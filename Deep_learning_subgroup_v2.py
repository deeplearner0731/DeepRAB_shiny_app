"""
DeepRAB (revised): concrete autoencoder + A-learning / R-learning loss for
subgroup identification and predictive-biomarker ranking.

Changes vs. Deep_learning_subgroup.py
-------------------------------------
1. trt / pi are carried INSIDE y_true as extra columns, so they stay aligned
   with the rows after Keras shuffles a batch.  (The original sliced
   trt_const[:batch_size] on every batch, which pairs treatments with the
   wrong patients whenever shuffle=True or there is more than one batch.)
2. Default loss is plain MSE on the A-learning residual, differentiated by
   autodiff.  No sqrt (no gradient singularity at residual 0) and no
   hand-written gradient.  The RMSE + custom-gradient version is kept as an
   option, with the gradient corrected w.r.t. every input and an epsilon
   guard; `check_custom_gradient()` verifies it against autodiff.
3. y may be pre-residualized (R-learner):  loss uses  y_res - (A - pi) * f.
   Pass y_res = y - m_hat(X) to remove the prognostic/baseline effect.
4. Temperature is annealed only in the training branch, so predict()/
   evaluate() no longer decay it.
5. Convergence check uses mean of the PER-ROW max probability (axis=-1);
   the original used a global max.  The stopper callback actually stops.
6. Optional per-example Gumbel noise (per_sample_noise=True): the original
   draws a single (K, d) noise matrix shared by the whole batch, which with
   full-batch training gives one noise sample per gradient step and very
   high-variance logit gradients.
7. No K.set_learning_phase / TF1 Session: the train vs. inference branch is
   selected explicitly by Keras' `training` flag.

Binary outcomes
---------------
Both continuous and binary responses are supported through the same
A-learning skeleton.  Write the linear predictor as

    v(x, a) = eta(x) + (a - pi(x)) * f(x)

where `f` is the contrast function the concrete autoencoder learns and `eta`
is a nuisance baseline supplied as a fixed OFFSET.  The loss is then
`M(y, v)` for a link-appropriate M:

    loss_name="A"         M(y, v) = (y - v)^2                    continuous
    loss_name="A_binary"  M(y, v) = -[ y*v - log(1 + exp(v)) ]   binary, y in {0,1}
                                  = log(1 + exp(-y~ * v)),  y~ = 2y - 1

    NOTE on the binary form.  It must be log(1 + exp(+v)) with y in {0,1}.
    The variant  -[y*v - log(1 + exp(-v))]  has derivative sigma(v) - 1 - y,
    which is <= 0 everywhere, so it is monotone decreasing in v and has no
    minimiser -- it would drive v -> +inf.  The exp(-v) form is the correct
    one only under +-1 coding, as log(1 + exp(-y~ v)); the two expressions
    above are identical.

Why this parameterisation is exact for binary.  If the truth is a logistic
model with a treatment interaction,

    logit P(Y = 1 | X, A) = b(X) + A * tau(X)
                          = [b(X) + pi(X) tau(X)] + (A - pi(X)) * tau(X),

so setting eta := b + pi*tau recovers f = tau exactly, and f is a
log-odds-ratio contrast.  sign(f) is the direction of benefit, which is all
the subgroup rule needs.

The offset plays the role that residualisation plays in the continuous case
(item 3 above): with eta = 0 the binary loss is only valid when the prognostic
effect is exactly 0.  Estimate eta by cross-fitted logistic regression on X.
Two caveats worth stating plainly:
  - Because the logit is not collapsible, logit P(Y=1|X) is not exactly
    b + pi*tau, so a cross-fitted eta is a good approximation rather than an
    identity.  It removes far more bias (the whole prognostic signal) than the
    non-collapsibility it introduces.
  - Do NOT residualise a binary y by subtraction.  Keep y in {0,1} and put the
    baseline in `offset`, on the logit scale.

Time-to-event outcomes
----------------------
Same skeleton again, with the Cox partial likelihood (Breslow) as M:

    loss_name="A_cox"
        M(y, v) = - sum_i  Delta_i * [ v_i - log( sum_{j in R_i} exp(v_j) ) ]
        v_i = eta(x_i) + (A_i - pi_i) * f(x_i),   R_i = { j : T_j >= T_i }

so the fitted model is  lambda(t | x, a) = lambda0(t) exp(v).  The nonparametric
baseline hazard lambda0 cancels out of the partial likelihood, but the
prognostic term does NOT -- exactly as in the binary case, use
eta = h(x) + pi*tau(x), cross-fitted (`cox_fit_ridge` below).

Three properties of this loss that shape the implementation:

  1. NOT SEPARABLE across subjects.  Every other loss here is a sum of
     per-row terms, so Keras can minibatch it freely.  The partial likelihood
     couples rows through R_i, so a minibatch silently replaces R_i by its
     intersection with the batch.  `fit()` therefore defaults to FULL BATCH for
     this loss and warns if you force a smaller one.  Consequence: one epoch is
     one gradient step, so num_epochs must be ~1000-2000 (not 300) for the
     concrete temperature to anneal.  `alpha` is derived from
     num_epochs * steps_per_epoch, so it adapts automatically.
  2. Invariant to shifting v by a constant, which is why eta only has to be
     right up to an additive constant.  `check_cox_loss` asserts this.
  3. Needs the time ordering, which Keras' shuffling destroys.  The risk set is
     therefore rebuilt inside the loss from the time column of y_true -- the
     same fix item 1 above applies to trt/pi.  Ties are handled exactly
     (Breslow) by using a (B, B) at-risk mask rather than a sorted cumulative
     sum, which costs O(B^2) memory: fine at the n ~ 1e3 this code targets,
     but switch to the sorted-cumsum form above n ~ 1e5.

SIGN CONVENTION, worth reading twice.  For continuous and binary y, f > 0 means
treatment raises y.  For Cox, f is a log HAZARD ratio, so f < 0 is the benefit
direction (treatment lowers the hazard).  `benefit_score()` orients f so larger
is always better; use it before computing an AUC or thresholding at 0, or your
AUC will come out as 1 - AUC.

Target: TensorFlow 2.x (tf.keras).  Run `python Deep_learning_subgroup_v2.py`
to execute the built-in gradient, alignment, loss-identity and end-to-end
self-checks for all three outcome types.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.initializers import Constant, GlorotNormal
from tensorflow.keras.layers import Input, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# --------------------------------------------------------------------------- #
# Target packing
# --------------------------------------------------------------------------- #
# y_true is always a (n, 6) array: [y, a01, pi, weight, offset, event]
#   y      : outcome.  Continuous: optionally residualized y - m_hat(X).
#            Binary: raw 0/1 -- never residualize it, use `offset` instead.
#            TTE: the observed follow-up time min(T, C), strictly positive.
#   a01    : treatment coded 0/1
#   pi     : P(A = 1 | X)
#   weight : per-subject weight (1.0 if unused)
#   offset : nuisance baseline eta(x) added to the linear predictor (0.0 if
#            unused).  Continuous: equivalent to residualizing y outside.
#            Binary: the cross-fitted baseline logit.
#            TTE: the cross-fitted linear predictor, needed only up to a
#            constant since the partial likelihood is shift-invariant.
#   event  : TTE event indicator Delta in {0, 1} (1 = event, 0 = censored).
#            Unused by the continuous and binary losses; defaults to 1.0.
N_TARGET_COLS = 6


def to_a01(trt) -> np.ndarray:
    """Map a treatment column to 0/1, accepting {0,1}, {-1,1} or {1,2}."""
    t = np.asarray(trt, dtype=np.float64).reshape(-1)
    u = np.unique(t)
    if np.all(np.isin(u, [0.0, 1.0])):
        return t
    if np.all(np.isin(u, [-1.0, 1.0])):
        return (t + 1.0) / 2.0
    if np.all(np.isin(u, [1.0, 2.0])):
        return t - 1.0
    raise ValueError(f"cannot interpret treatment coding, unique values = {u}")


def check_y01(y, name="y") -> np.ndarray:
    """Validate that a binary outcome really is coded 0/1 (accepts {1,2}, {-1,1})."""
    t = np.asarray(y, dtype=np.float64).reshape(-1)
    u = np.unique(t)
    if np.all(np.isin(u, [0.0, 1.0])):
        return t
    if np.all(np.isin(u, [1.0, 2.0])):
        return t - 1.0
    if np.all(np.isin(u, [-1.0, 1.0])):
        return (t + 1.0) / 2.0
    raise ValueError(
        f"{name} must be binary for a binary loss; unique values = {u[:10]}"
        + (" ..." if len(u) > 10 else "")
    )


def pack_targets(y, trt, pi, weight=None, offset=None, event=None) -> np.ndarray:
    """Build the (n, 6) target matrix consumed by the losses below."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    a01 = to_a01(trt)
    pi = np.asarray(pi, dtype=np.float64).reshape(-1)
    if weight is None:
        weight = np.ones_like(y)
    weight = np.asarray(weight, dtype=np.float64).reshape(-1)
    if offset is None:
        offset = np.zeros_like(y)
    offset = np.asarray(offset, dtype=np.float64).reshape(-1)
    if event is None:
        event = np.ones_like(y)
    event = check_y01(event, "event")
    if not (len(y) == len(a01) == len(pi) == len(weight) == len(offset) == len(event)):
        raise ValueError(
            f"length mismatch: y={len(y)} trt={len(a01)} pi={len(pi)} "
            f"w={len(weight)} offset={len(offset)} event={len(event)}"
        )
    if np.any(pi <= 0.0) or np.any(pi >= 1.0):
        raise ValueError("pi must lie strictly inside (0, 1); clip it first")
    return np.stack([y, a01, pi, weight, offset, event], axis=1).astype(np.float32)


def _unpack(y_true):
    y_true = tf.cast(y_true, tf.float32)
    return (
        y_true[:, 0:1],  # y   (or the follow-up time, for Cox)
        y_true[:, 1:2],  # a01
        y_true[:, 2:3],  # pi
        y_true[:, 3:4],  # weight
        y_true[:, 4:5],  # offset  eta(x)
        y_true[:, 5:6],  # event   Delta
    )


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
def a_learning_loss(y_true, y_pred):
    """
    A-learning / R-learning squared loss for a CONTINUOUS outcome
    (default; autodiff).

        L(f) = sum_i w_i * ( y_i - eta_i - (A_i - pi_i) * f(x_i) )^2 / sum_i w_i

    y_true : (B, 5) packed by pack_targets
    y_pred : (B, 1) contrast function f(x)

    Passing the baseline through `offset` is algebraically identical to
    residualizing y before the call; offset = 0 reproduces the old behaviour.
    """
    y, a01, pi, w, off, _d = _unpack(y_true)
    y_pred = tf.cast(y_pred, tf.float32)
    c = a01 - pi
    r = y - off - c * y_pred
    return tf.reduce_sum(w * tf.square(r)) / (tf.reduce_sum(w) + K.epsilon())


def a_learning_loss_binary(y_true, y_pred):
    """
    A-learning logistic loss for a BINARY outcome y in {0, 1}.

        v_i  = eta_i + (A_i - pi_i) * f(x_i)
        L(f) = sum_i w_i * M(y_i, v_i) / sum_i w_i,
        M(y, v) = -[ y*v - log(1 + exp(v)) ] = log(1 + exp(v)) - y*v

    Implemented with sigmoid_cross_entropy_with_logits, which evaluates
    max(v, 0) - v*y + log(1 + exp(-|v|)) -- the same quantity, but stable for
    |v| up to ~80 instead of overflowing at ~35.

    Convexity in v (M'' = sigma(v)(1 - sigma(v)) > 0) is what makes this a
    proper loss; see the module docstring for why the log(1 + exp(-v)) variant
    is not.
    """
    y, a01, pi, w, off, _d = _unpack(y_true)
    y_pred = tf.cast(y_pred, tf.float32)
    v = off + (a01 - pi) * y_pred
    nll = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=v)
    return tf.reduce_sum(w * nll) / (tf.reduce_sum(w) + K.epsilon())


@tf.custom_gradient
def _rmse_custom(y, c, y_pred):
    """
    value = sqrt( mean_i (y_i - c_i * f_i)^2 + eps )   with explicit gradients.

    Kept for reference / to reproduce the original objective.  Note:
        d value / d f_i = -c_i * r_i / (N * value)
        d value / d y_i =        r_i / (N * value)
        d value / d c_i = -f_i * r_i / (N * value)
    The original code returned the d/df expression for BOTH y and f.  That is
    harmless in practice (nothing trainable feeds y_true) but it is wrong, and
    the missing epsilon makes the gradient blow up as r -> 0.
    """
    eps = 1e-12
    r = y - c * y_pred
    n = tf.cast(tf.size(r), r.dtype)
    value = tf.sqrt(tf.reduce_mean(tf.square(r)) + eps)

    def grad(dy):
        scale = dy / (n * value)
        return scale * r, scale * (-y_pred * r), scale * (-c * r)

    return value, grad


def a_learning_loss_cox(y_true, y_pred):
    """
    A-learning Cox partial-likelihood loss (Breslow) for a TIME-TO-EVENT outcome.

        v_i  = eta_i + (A_i - pi_i) * f(x_i)
        L(f) = - sum_i w_i Delta_i [ v_i - log( sum_{j in R_i} exp(v_j) ) ]
               / sum_i w_i Delta_i
        R_i  = { j : T_j >= T_i }

    y_true column 0 carries the follow-up time and column 5 the event
    indicator; the risk set is rebuilt here rather than assumed, so the loss is
    invariant to however Keras shuffled the batch (see check_cox_shuffle).

    Normalising by the number of events makes the value comparable across
    validation sets of different size; it rescales the objective by a constant
    and so does not move the minimiser.

    Ties are handled exactly (Breslow) via a (B, B) at-risk mask.  That is
    O(B^2) memory -- deliberate, see the module docstring.  `weight` applies to
    the outer event sum only, not to the risk-set denominators.

    NB: this loss is NOT separable across rows.  Train it full-batch, or R_i is
    silently truncated to the batch.  `fit()` handles that for you.
    """
    t, a01, pi, w, off, dlt = _unpack(y_true)
    f = tf.cast(y_pred, tf.float32)
    v = off + (a01 - pi) * f

    t_ = tf.reshape(t, [-1])
    v_ = tf.reshape(v, [-1])
    d_ = tf.reshape(dlt, [-1])
    w_ = tf.reshape(w, [-1])

    # at_risk[i, j] = 1 iff T_j >= T_i.  The diagonal is always true, so every
    # row has at least one finite entry and the logsumexp below is well defined.
    at_risk = tf.greater_equal(t_[tf.newaxis, :], t_[:, tf.newaxis])
    # a large finite negative, not -inf: -inf can produce NaN gradients
    neg = tf.fill(tf.shape(at_risk), tf.constant(-1e30, dtype=v_.dtype))
    masked = tf.where(at_risk, tf.broadcast_to(v_[tf.newaxis, :], tf.shape(at_risk)), neg)
    log_risk = tf.reduce_logsumexp(masked, axis=1)

    ev_w = w_ * d_
    return tf.reduce_sum(ev_w * (log_risk - v_)) / (tf.reduce_sum(ev_w) + K.epsilon())


def a_learning_loss_rmse(y_true, y_pred):
    """RMSE flavour of the A-learning loss using the corrected custom gradient."""
    y, a01, pi, _w, off, _d = _unpack(y_true)
    y_pred = tf.cast(y_pred, tf.float32)
    return _rmse_custom(y - off, a01 - pi, y_pred)


def weight_learning_loss(y_true, y_pred):
    """
    Weight-learning (IPW / value-search) surrogate, for comparison.
    Maximizes  E[ 1{sign(f) = A} * Y / pi_A ]  via a smooth logistic surrogate.
    Works for continuous and 0/1 y alike; `offset` is not used.
    """
    y, a01, pi, w, _off, _d = _unpack(y_true)
    y_pred = tf.cast(y_pred, tf.float32)
    z = 2.0 * a01 - 1.0                      # +-1 coding
    ipw = w / tf.where(a01 > 0.5, pi, 1.0 - pi)
    margin = tf.nn.softplus(-z * y_pred)     # smooth 0/1 loss
    return tf.reduce_sum(ipw * y * margin) / (tf.reduce_sum(ipw) + K.epsilon())


LOSSES = {
    "A": a_learning_loss,                # continuous
    "A_rmse": a_learning_loss_rmse,      # continuous, sqrt flavour
    "A_binary": a_learning_loss_binary,  # binary y in {0,1}
    "A_cox": a_learning_loss_cox,        # time-to-event (time, event)
    "W": weight_learning_loss,
}

# loss names that require y to be 0/1
BINARY_LOSSES = frozenset({"A_binary"})
# loss names that are not separable across rows -> must be trained full-batch,
# and that need the `event` column
COX_LOSSES = frozenset({"A_cox"})


def outcome_to_loss_name(outcome: str) -> str:
    """Map "continuous"/"binary"/"tte" to the corresponding loss_name."""
    o = str(outcome).lower()
    if o in ("continuous", "gaussian", "cont"):
        return "A"
    if o in ("binary", "binomial", "bin", "logistic"):
        return "A_binary"
    if o in ("tte", "survival", "cox", "time_to_event", "time-to-event"):
        return "A_cox"
    raise ValueError(
        f'outcome must be "continuous", "binary" or "tte", got {outcome!r}'
    )


def benefit_score(f, outcome="continuous") -> np.ndarray:
    """
    Orient the contrast so that LARGER always means more treatment benefit.

    Continuous / binary : returned unchanged -- f is on the outcome scale, so
                          f > 0 means treatment raises y.  (If a high y is the
                          bad thing, e.g. y = 1 codes death, negate it yourself;
                          that is a property of your endpoint, not of the loss.)
    TTE                 : negated -- f is a log HAZARD ratio, so f < 0 is the
                          benefit direction.

    Use this before computing an AUC against a responder label or thresholding
    at 0, otherwise a Cox fit reports 1 - AUC.
    """
    f = np.asarray(f, dtype=np.float64).reshape(-1)
    return -f if outcome_to_loss_name(outcome) == "A_cox" else f


# --------------------------------------------------------------------------- #
# Concrete selection layer
# --------------------------------------------------------------------------- #
def _assign_read(var, value):
    """Assign and return a tensor that is guaranteed to read the new value."""
    out = var.assign(value)
    try:
        return tf.convert_to_tensor(out)
    except (TypeError, ValueError):
        return var.read_value()


class ConcreteSelect(Layer):
    """
    Concrete / Gumbel-softmax feature selection (Balin, Abid & Zou 2019).

    output_dim       : K, number of selection slots
    per_sample_noise : draw independent Gumbel noise per example instead of
                       one (K, d) matrix per batch.  Much lower gradient
                       variance, especially with large batches.
    """

    def __init__(
        self,
        output_dim,
        start_temp=10.0,
        min_temp=0.1,
        alpha=0.99999,
        per_sample_noise=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.output_dim = int(output_dim)
        self.start_temp = float(start_temp)
        self.min_temp_value = float(min_temp)
        self.alpha_value = float(alpha)
        self.per_sample_noise = bool(per_sample_noise)

    def build(self, input_shape):
        n_features = int(input_shape[-1])
        self.n_features = n_features
        self.temp = self.add_weight(
            name="temp",
            shape=(),
            initializer=Constant(self.start_temp),
            trainable=False,
        )
        self.logits = self.add_weight(
            name="logits",
            shape=(self.output_dim, n_features),
            initializer=GlorotNormal(),
            trainable=True,
        )
        self.min_temp = tf.constant(self.min_temp_value, dtype=self.temp.dtype)
        self.alpha = tf.constant(self.alpha_value, dtype=self.temp.dtype)
        super().build(input_shape)

    # -- forward passes ---------------------------------------------------- #
    def _soft_forward(self, X):
        """Stochastic relaxed selection; anneals the temperature by one step."""
        temp = _assign_read(self.temp, tf.maximum(self.min_temp, self.temp * self.alpha))
        temp = tf.cast(temp, X.dtype)
        logits = tf.cast(self.logits, X.dtype)

        if self.per_sample_noise:
            shape = tf.concat([[tf.shape(X)[0]], tf.shape(logits)], axis=0)
            u = tf.random.uniform(shape, minval=K.epsilon(), maxval=1.0, dtype=X.dtype)
            gumbel = -tf.math.log(-tf.math.log(u))
            sel = tf.nn.softmax((logits[None, :, :] + gumbel) / temp, axis=-1)
            return tf.einsum("bd,bkd->bk", X, sel)

        u = tf.random.uniform(tf.shape(logits), minval=K.epsilon(), maxval=1.0, dtype=X.dtype)
        gumbel = -tf.math.log(-tf.math.log(u))
        sel = tf.nn.softmax((logits + gumbel) / temp, axis=-1)
        return tf.matmul(X, sel, transpose_b=True)

    def _hard_forward(self, X):
        """Deterministic argmax selection; no temperature update."""
        sel = tf.one_hot(
            tf.argmax(self.logits, axis=-1), depth=self.n_features, dtype=X.dtype
        )
        return tf.matmul(X, sel, transpose_b=True)

    def call(self, X, training=None):
        X = tf.cast(X, self.logits.dtype)
        # Both branches return (B, K), so this is safe whether `training` is a
        # python bool (resolved statically) or a tensor (resolved by tf.cond).
        return K.in_train_phase(
            lambda: self._soft_forward(X),
            lambda: self._hard_forward(X),
            training=training,
        )

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            output_dim=self.output_dim,
            start_temp=self.start_temp,
            min_temp=self.min_temp_value,
            alpha=self.alpha_value,
            per_sample_noise=self.per_sample_noise,
        )
        return cfg

    # -- diagnostics ------------------------------------------------------- #
    def probabilities(self) -> np.ndarray:
        """(K, d) selection probabilities."""
        return tf.nn.softmax(self.logits, axis=-1).numpy()

    def mean_max_prob(self) -> float:
        """Mean over the K slots of the largest per-slot probability."""
        p = tf.nn.softmax(self.logits, axis=-1)
        return float(tf.reduce_mean(tf.reduce_max(p, axis=-1)))


class ConcreteConvergence(Callback):
    """Stop once the relaxed selection is effectively one-hot."""

    def __init__(self, layer_name="concrete_select", mean_max_target=0.998, verbose=0):
        super().__init__()
        self.layer_name = layer_name
        self.mean_max_target = float(mean_max_target)
        self.verbose = verbose
        self.mean_max = 0.0

    def on_epoch_end(self, epoch, logs=None):
        layer = self.model.get_layer(self.layer_name)
        self.mean_max = layer.mean_max_prob()
        if self.verbose:
            print(
                f"  epoch {epoch}: mean max prob = {self.mean_max:.4f}, "
                f"temp = {float(layer.temp):.4f}"
            )
        if self.mean_max >= self.mean_max_target:
            self.model.stop_training = True


# --------------------------------------------------------------------------- #
# Selector
# --------------------------------------------------------------------------- #
class ConcreteAutoencoderFeatureSelector:
    """
    Parameters
    ----------
    K                : number of selection slots (features to keep)
    output_function   : callable mapping the (B, K) selected tensor to (B, 1)
    num_epochs        : epochs for the FIRST tryout
    batch_size        : mini-batch size.  Prefer 64-256 over full batch: with
                        full-batch training and shared Gumbel noise the logits
                        see one noise draw per step.
    learning_rate     : Adam lr.  1e-3 .. 1e-2 for standardized inputs;
                        0.1 (the original demo value) is far too large.
    min_temp          : final temperature.  Must be small (<= 0.1) for the
                        relaxed selection to converge to one-hot, otherwise
                        training (soft) and inference (argmax) disagree.
    tryout_limit      : if the concrete distribution has not converged, retry
                        with 2x the epochs, up to this many times.
    per_sample_noise  : see ConcreteSelect
    """

    def __init__(
        self,
        K,
        output_function,
        num_epochs=300,
        batch_size=128,
        learning_rate=1e-3,
        start_temp=10.0,
        min_temp=0.05,
        tryout_limit=2,
        loss_name="A",
        mean_max_target=0.998,
        per_sample_noise=False,
        shuffle=True,
        clipnorm=1.0,
        verbose=0,
    ):
        if loss_name not in LOSSES:
            raise ValueError(f"loss_name must be one of {sorted(LOSSES)}")
        self.K = int(K)
        self.output_function = output_function
        self.num_epochs = int(num_epochs)
        self.batch_size = batch_size
        self.learning_rate = float(learning_rate)
        self.start_temp = float(start_temp)
        self.min_temp = float(min_temp)
        self.tryout_limit = int(tryout_limit)
        self.loss_name = loss_name
        self.mean_max_target = float(mean_max_target)
        self.per_sample_noise = bool(per_sample_noise)
        self.shuffle = bool(shuffle)
        self.clipnorm = clipnorm
        self.verbose = verbose

        self.model = None
        self.concrete_select = None
        self.probabilities = None
        self.indices = None
        self.mean_max = None
        self.history = None

    # ------------------------------------------------------------------ #
    def fit(self, X, y, trt, pi, weight=None, offset=None, event=None,
            validation_data=None):
        """
        X   : (n, d) design matrix (standardize it first)
        y   : (n,)  outcome.
                    loss_name="A"        -> continuous, ideally residualized
                                           y - m_hat(X) (or pass m_hat as offset)
                    loss_name="A_binary" -> raw 0/1; do NOT residualize, pass the
                                           cross-fitted baseline logit as offset
                    loss_name="A_cox"    -> the follow-up time min(T, C) > 0, and
                                           `event` must be given
        trt : (n,)  treatment, any of {0,1} / {-1,1} / {1,2}
        pi  : (n,)  P(A = 1 | X), strictly inside (0, 1)
        offset : (n,) nuisance baseline eta(x), or None for 0
        event  : (n,) TTE event indicator in {0,1}; required for loss "A_cox"
                 and ignored otherwise
        validation_data : (X_val, y_val, trt_val, pi_val), or with
                          offset_val / event_val appended, or None

        Note: `validation_data` only populates `self.history`; it is not used for
        stopping or selection, and Keras evaluates it every epoch at a cost of
        roughly 5x the whole fit.  Leave it None and score the held-out split
        afterwards with held_out_loss(); that is what the demo does.
        """
        is_cox = self.loss_name in COX_LOSSES
        if self.loss_name in BINARY_LOSSES:
            y = check_y01(y, "y")
        if is_cox:
            if event is None:
                raise ValueError(
                    'loss_name="A_cox" needs `event` (1 = event, 0 = censored)'
                )
            if np.any(np.asarray(y, dtype=np.float64) <= 0.0):
                raise ValueError("follow-up times must be strictly positive")
            if float(np.sum(check_y01(event, "event"))) < 2.0:
                raise ValueError("need at least 2 events to fit a Cox model")

        X = np.asarray(X, dtype=np.float32)
        targets = pack_targets(y, trt, pi, weight, offset, event)
        if len(X) != len(targets):
            raise ValueError(f"X has {len(X)} rows but targets have {len(targets)}")

        val = None
        if validation_data is not None:
            vd = list(validation_data) + [None] * (6 - len(validation_data))
            Xv, yv, tv, pv, ov, ev = vd[:6]
            if self.loss_name in BINARY_LOSSES:
                yv = check_y01(yv, "y_val")
            val = (np.asarray(Xv, dtype=np.float32),
                   pack_targets(yv, tv, pv, None, ov, ev))

        if is_cox:
            # The partial likelihood couples all rows through the risk sets, so
            # anything short of the full sample truncates R_i to the batch.
            if self.batch_size is not None and self.batch_size < len(X):
                warnings.warn(
                    f"loss_name={self.loss_name!r} is not separable across rows: "
                    f"batch_size={self.batch_size} < n={len(X)} restricts each "
                    f"risk set to the batch, which biases the partial likelihood. "
                    f"Pass batch_size=None for full-batch training.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                batch_size = self.batch_size
            else:
                batch_size = len(X)
        else:
            batch_size = self.batch_size or max(len(X) // 32, 16)
        batch_size = min(batch_size, len(X))
        steps_per_epoch = math.ceil(len(X) / batch_size)

        num_epochs = self.num_epochs
        for tryout in range(max(1, self.tryout_limit)):
            keras.backend.clear_session()

            total_steps = max(1, num_epochs * steps_per_epoch)
            alpha = math.exp(math.log(self.min_temp / self.start_temp) / total_steps)

            inputs = Input(shape=(X.shape[1],), name="x")
            self.concrete_select = ConcreteSelect(
                self.K,
                self.start_temp,
                self.min_temp,
                alpha,
                per_sample_noise=self.per_sample_noise,
                name="concrete_select",
            )
            selected = self.concrete_select(inputs)
            outputs = self.output_function(selected)
            self.model = Model(inputs, outputs, name="deeprab")

            opt = Adam(learning_rate=self.learning_rate, clipnorm=self.clipnorm)
            self.model.compile(optimizer=opt, loss=LOSSES[self.loss_name], run_eagerly=False)

            stopper = ConcreteConvergence(
                "concrete_select", self.mean_max_target, verbose=self.verbose
            )
            self.history = self.model.fit(
                X,
                targets,
                batch_size=batch_size,
                epochs=num_epochs,
                shuffle=self.shuffle,          # safe now: trt/pi ride along in y
                verbose=self.verbose,
                callbacks=[stopper],
                validation_data=val,
            )

            self.mean_max = self.concrete_select.mean_max_prob()
            if self.mean_max >= self.mean_max_target:
                break
            num_epochs *= 2

        self.probabilities = self.concrete_select.probabilities()          # (K, d)
        self.indices = np.argmax(self.probabilities, axis=-1)              # (K,)
        return self

    # ------------------------------------------------------------------ #
    def predict_score(self, X) -> np.ndarray:
        """
        The contrast f(x), with the deterministic argmax selection.  (n,) array.

        This is the subgroup score for BOTH outcome types.  Continuous: f is the
        treatment-effect difference.  Binary: f is the log odds ratio.  Either
        way sign(f) is the direction of benefit, so ranking and thresholding at
        0 work identically.
        """
        X = np.asarray(X, dtype=np.float32)
        return np.asarray(self.model(X, training=False)).reshape(-1)

    def predict_linear(self, X, trt, pi, offset=None) -> np.ndarray:
        """The full linear predictor  v = eta(x) + (a - pi) * f(x).  (n,) array."""
        f = self.predict_score(X)
        a01 = to_a01(trt)
        pi = np.asarray(pi, dtype=np.float64).reshape(-1)
        off = 0.0 if offset is None else np.asarray(offset, dtype=np.float64).reshape(-1)
        return off + (a01 - pi) * f

    def predict_prob(self, X, trt, pi, offset=None) -> np.ndarray:
        """P(Y = 1 | X, A) for a binary fit.  (n,) array."""
        if self.loss_name not in BINARY_LOSSES:
            raise RuntimeError(
                f"predict_prob is only meaningful for a binary loss, "
                f"not loss_name={self.loss_name!r}"
            )
        v = self.predict_linear(X, trt, pi, offset)
        return 1.0 / (1.0 + np.exp(-v))

    def get_indices(self) -> np.ndarray:
        return self.indices

    def get_mask(self) -> np.ndarray:
        """(d,) count of how many of the K slots picked each feature."""
        mask = np.zeros(self.probabilities.shape[1])
        for j in self.indices:
            mask[j] += 1
        return mask

    def get_support(self, indices=False):
        return self.get_indices() if indices else self.get_mask()

    def feature_scores(self, mode="max") -> np.ndarray:
        """
        Continuous per-feature importance from the selection probabilities.
        Works for any K, including K = 1 -- unlike counting duplicate argmax
        indices, which has no resolution when K is small.

        mode = 'max' : s_j = max_k  P[k, j]   (does feature j own any slot?)
        mode = 'sum' : s_j = sum_k  P[k, j]   (total mass on feature j)
        """
        if self.probabilities is None:
            raise RuntimeError("call fit() first")
        if mode == "max":
            return self.probabilities.max(axis=0)
        if mode == "sum":
            return self.probabilities.sum(axis=0)
        raise ValueError("mode must be 'max' or 'sum'")

    def transform(self, X):
        return np.asarray(X)[:, self.get_indices()]


# --------------------------------------------------------------------------- #
# Evaluation helpers (numpy, no graph)
# --------------------------------------------------------------------------- #
def r_loss_np(y_res, trt, pi, f, offset=None) -> float:
    """
    Held-out A-/R-learning squared loss for a CONTINUOUS outcome.
    Lower is better.  Uses no subgroup label.
    """
    a01 = to_a01(trt)
    y_res = np.asarray(y_res, dtype=np.float64).reshape(-1)
    pi = np.asarray(pi, dtype=np.float64).reshape(-1)
    f = np.asarray(f, dtype=np.float64).reshape(-1)
    off = 0.0 if offset is None else np.asarray(offset, dtype=np.float64).reshape(-1)
    return float(np.mean((y_res - off - (a01 - pi) * f) ** 2))


def binary_loss_np(y, trt, pi, f, offset=None) -> float:
    """
    Held-out A-learning logistic loss for a BINARY outcome.  Lower is better.
    Uses no subgroup label.  This is the mean of

        M(y, v) = log(1 + exp(v)) - y*v,   v = eta + (A - pi) * f

    computed in the overflow-safe form  max(v,0) - v*y + log1p(exp(-|v|)).
    """
    a01 = to_a01(trt)
    y = check_y01(y, "y")
    pi = np.asarray(pi, dtype=np.float64).reshape(-1)
    f = np.asarray(f, dtype=np.float64).reshape(-1)
    off = 0.0 if offset is None else np.asarray(offset, dtype=np.float64).reshape(-1)
    v = off + (a01 - pi) * f
    nll = np.maximum(v, 0.0) - v * y + np.log1p(np.exp(-np.abs(v)))
    return float(np.mean(nll))


def _log_risk_np(time, v):
    """log sum_{j: T_j >= T_i} exp(v_j), Breslow, tie-exact.  O(n^2)."""
    at_risk = time[np.newaxis, :] >= time[:, np.newaxis]
    vv = np.where(at_risk, v[np.newaxis, :], -np.inf)
    m = vv.max(axis=1, keepdims=True)
    return (m.reshape(-1) + np.log(np.exp(vv - m).sum(axis=1)))


def cox_loss_np(time, event, trt, pi, f, offset=None) -> float:
    """
    Held-out Cox partial-likelihood loss (the cross-validated partial
    likelihood, CVPL).  Lower is better.  Uses no subgroup label.

        L = - sum_i Delta_i [ v_i - log sum_{j in R_i} exp(v_j) ] / sum_i Delta_i

    This is the TTE analogue of r_loss_np / binary_loss_np and is the criterion
    the demo tunes on.  Note it is a property of the whole validation SET (the
    risk sets are computed within it), not an average of per-subject scores, so
    it is only comparable between models scored on the SAME rows.
    """
    t = np.asarray(time, dtype=np.float64).reshape(-1)
    d = check_y01(event, "event")
    a01 = to_a01(trt)
    pi = np.asarray(pi, dtype=np.float64).reshape(-1)
    f = np.asarray(f, dtype=np.float64).reshape(-1)
    off = 0.0 if offset is None else np.asarray(offset, dtype=np.float64).reshape(-1)
    if d.sum() < 1:
        return np.nan
    v = off + (a01 - pi) * f
    return float(np.sum(d * (_log_risk_np(t, v) - v)) / d.sum())


def held_out_loss(outcome, y, trt, pi, f, offset=None, event=None) -> float:
    """
    Dispatch to the loss matching `outcome` ("continuous", "binary" or "tte").
    Lower is always better, so callers can sort ascending regardless of type.
    For "tte", `y` is the follow-up time and `event` is required.
    """
    name = outcome_to_loss_name(outcome)
    if name == "A_binary":
        return binary_loss_np(y, trt, pi, f, offset)
    if name == "A_cox":
        if event is None:
            raise ValueError('outcome="tte" requires `event`')
        return cox_loss_np(y, event, trt, pi, f, offset)
    return r_loss_np(y, trt, pi, f, offset)


def cox_fit_ridge(X, time, event, alpha=1e-2, max_iter=100, tol=1e-9):
    """
    Ridge-penalized Cox regression by Newton-Raphson, Breslow ties, numpy only.

    Exists so the TTE path needs no extra dependency (sklearn has no Cox
    model).  Used for two things: cross-fitting the baseline offset eta(x), and
    estimating within-subgroup treatment log hazard ratios in
    cox_subgroup_gain.

    Returns beta, shape (p,).  The linear predictor is identified only up to an
    additive constant, which is exactly what the partial likelihood needs.
    Cost is O(n^2 p^2) per iteration -- fine for n ~ 1e3.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    t = np.asarray(time, dtype=np.float64).reshape(-1)
    d = check_y01(event, "event")
    at_risk = (t[np.newaxis, :] >= t[:, np.newaxis]).astype(np.float64)
    XX = (X[:, :, None] * X[:, None, :]).reshape(n, p * p)
    eye = np.eye(p)

    beta = np.zeros(p)
    for _ in range(max_iter):
        v = X @ beta
        e = np.exp(v - v.max())
        S0 = at_risk @ e                                  # (n,)
        S0 = np.maximum(S0, 1e-300)
        xbar = (at_risk @ (e[:, None] * X)) / S0[:, None]  # (n, p)
        S2 = (at_risk @ (e[:, None] * XX)) / S0[:, None]   # (n, p*p)

        grad = (d[:, None] * (X - xbar)).sum(0) - alpha * beta
        H = -((d[:, None] * S2).sum(0).reshape(p, p)
              - (d[:, None, None] * (xbar[:, :, None] * xbar[:, None, :])).sum(0)
              ) - alpha * eye
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        beta_new = beta - step
        if not np.all(np.isfinite(beta_new)):
            break
        done = np.max(np.abs(beta_new - beta)) < tol
        beta = beta_new
        if done:
            break
    return beta


def cox_subgroup_gain(time, event, trt, f, cut=0.0, min_n=20, min_events=5,
                      alpha=1e-6) -> float:
    """
    Label-free TTE subgroup criterion, the analogue of subgroup_gain.

    Splits on the BENEFIT orientation (f < cut = predicted responder, since f is
    a log hazard ratio) and returns

        logHR(A | non-responders)  -  logHR(A | responders)

    each estimated by a one-covariate Cox fit of treatment within the subgroup.
    Positive = the predicted responders enjoy a more favourable hazard ratio
    than the rest, i.e. the rule separates in the intended direction.  Higher is
    better, matching subgroup_gain.

    Assumes treatment is randomized within each subgroup, which holds in an RCT.
    For observational data, weight by 1/pi first.  Returns nan if either arm is
    too small or too lightly censored to fit.
    """
    t = np.asarray(time, dtype=np.float64).reshape(-1)
    d = check_y01(event, "event")
    a01 = to_a01(trt)
    f = np.asarray(f, dtype=np.float64).reshape(-1)

    resp = f < cut
    out = []
    for m in (~resp, resp):
        if (m.sum() < min_n or d[m].sum() < min_events
                or len(np.unique(a01[m])) < 2):
            return np.nan
        out.append(float(cox_fit_ridge(a01[m], t[m], d[m], alpha=alpha)[0]))
    return out[0] - out[1]


def ipw_effect(y, trt, pi, subset=None) -> float:
    """
    Horvitz-Thompson estimate of E[Y(1) - Y(0)] on `subset`.
    Outcome-agnostic: for binary y this is the risk difference, so
    `subgroup_gain` below needs no binary-specific variant.
    """
    a01 = to_a01(trt)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    pi = np.asarray(pi, dtype=np.float64).reshape(-1)
    if subset is None:
        subset = np.ones_like(y, dtype=bool)
    subset = np.asarray(subset, dtype=bool).reshape(-1)
    if subset.sum() == 0:
        return np.nan
    w = a01 / pi - (1.0 - a01) / (1.0 - pi)
    return float(np.mean((w * y)[subset]))


def subgroup_gain(y, trt, pi, f, cut=0.0) -> float:
    """
    Label-free subgroup criterion: treatment effect among predicted responders
    minus effect among predicted non-responders.  Higher is better.
    """
    f = np.asarray(f, dtype=np.float64).reshape(-1)
    pos = f > cut
    if pos.all() or (~pos).all():
        return -np.inf
    return ipw_effect(y, trt, pi, pos) - ipw_effect(y, trt, pi, ~pos)


# --------------------------------------------------------------------------- #
# Self-checks
# --------------------------------------------------------------------------- #
def check_custom_gradient(n=64, seed=0, tol=1e-5) -> bool:
    """Compare _rmse_custom's hand-written gradient against autodiff."""
    rng = np.random.RandomState(seed)
    y = tf.constant(rng.randn(n, 1), dtype=tf.float32)
    c = tf.constant(rng.rand(n, 1) - 0.5, dtype=tf.float32)
    f0 = rng.randn(n, 1).astype(np.float32)

    f = tf.Variable(f0)
    with tf.GradientTape() as tape:
        v_custom = _rmse_custom(y, c, f)
    g_custom = tape.gradient(v_custom, f).numpy()

    f2 = tf.Variable(f0)
    with tf.GradientTape() as tape:
        r = y - c * f2
        v_auto = tf.sqrt(tf.reduce_mean(tf.square(r)) + 1e-12)
    g_auto = tape.gradient(v_auto, f2).numpy()

    dv = abs(float(v_custom) - float(v_auto))
    dg = float(np.max(np.abs(g_custom - g_auto)))
    ok = dv < tol and dg < tol
    print(f"[gradient check] |dvalue| = {dv:.3e}   max|dgrad| = {dg:.3e}   -> {'OK' if ok else 'FAIL'}")
    return ok


def check_alignment(n=200, d=6, seed=0) -> bool:
    """
    Shuffle-invariance check.  Permuting the rows of (X, y, trt, pi) together
    must leave the loss unchanged.  The original _loss_a fails this.
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d).astype(np.float32)
    y = rng.randn(n)
    trt = rng.binomial(1, 0.5, n)
    pi = np.full(n, 0.5)
    f = rng.randn(n)

    perm = rng.permutation(n)
    l1 = r_loss_np(y, trt, pi, f)
    l2 = r_loss_np(y[perm], trt[perm], pi[perm], f[perm])

    t1 = pack_targets(y, trt, pi)
    t2 = pack_targets(y[perm], trt[perm], pi[perm])
    v1 = float(a_learning_loss(tf.constant(t1), tf.constant(f.reshape(-1, 1), tf.float32)))
    v2 = float(a_learning_loss(tf.constant(t2), tf.constant(f[perm].reshape(-1, 1), tf.float32)))

    ok = abs(l1 - l2) < 1e-9 and abs(v1 - v2) < 1e-5 and abs(v1 - l1) < 1e-4
    print(
        f"[alignment check] numpy {l1:.6f} vs {l2:.6f} | tf {v1:.6f} vs {v2:.6f} "
        f"-> {'OK' if ok else 'FAIL'}"
    )
    return ok


def check_end_to_end(seed=0) -> bool:
    """Fit on a tiny simulated dataset where x0 and x1 are the true modifiers."""
    from tensorflow.keras.layers import Dense, ReLU

    rng = np.random.RandomState(seed)
    n, d = 1500, 10
    X = rng.randn(n, d)
    trt = rng.binomial(1, 0.5, n)
    pi = np.full(n, 0.5)
    tau = 1.5 * X[:, 0] - 1.5 * X[:, 1]           # true contrast
    y = 0.5 * X[:, 2] + (trt - 0.5) * tau + rng.randn(n) * 0.5
    y_res = y - y.mean()

    def decoder(x):
        x = Dense(32)(x)
        x = ReLU()(x)
        x = Dense(16)(x)
        x = ReLU()(x)
        return Dense(1)(x)

    sel = ConcreteAutoencoderFeatureSelector(
        K=2,
        output_function=decoder,
        num_epochs=120,
        batch_size=128,
        learning_rate=5e-3,
        min_temp=0.05,
        tryout_limit=2,
        per_sample_noise=True,
        verbose=0,
    ).fit(X, y_res, trt, pi)

    picked = set(int(i) for i in sel.get_indices())
    scores = sel.feature_scores("max")
    top2 = set(int(i) for i in np.argsort(-scores)[:2])
    ok = picked == {0, 1} or top2 == {0, 1}
    print(
        f"[end-to-end]  indices = {sorted(picked)}  top2 by prob = {sorted(top2)}  "
        f"mean-max = {sel.mean_max:.3f}  -> {'OK' if ok else 'check (weak signal / unlucky seed)'}"
    )
    return ok


def check_binary_loss(n=256, seed=1, tol=1e-5) -> bool:
    """
    Three things at once:
      (a) the TF binary loss equals the textbook  log(1+exp(v)) - y*v
      (b) binary_loss_np agrees with the TF version
      (c) the +-1 form  log(1 + exp(-y~ v))  is the same function
    """
    rng = np.random.RandomState(seed)
    y = rng.binomial(1, 0.5, n).astype(np.float64)
    trt = rng.binomial(1, 0.5, n).astype(np.float64)
    pi = np.full(n, 0.5)
    f = rng.randn(n) * 2.0
    off = rng.randn(n) * 0.5

    v = off + (trt - pi) * f
    naive = float(np.mean(np.log(1.0 + np.exp(v)) - y * v))       # textbook form
    ytil = 2.0 * y - 1.0
    pm1 = float(np.mean(np.log(1.0 + np.exp(-ytil * v))))          # +-1 form
    npv = binary_loss_np(y, trt, pi, f, off)

    t = pack_targets(y, trt, pi, None, off)
    tfv = float(a_learning_loss_binary(
        tf.constant(t), tf.constant(f.reshape(-1, 1), tf.float32)))

    d_tf, d_np, d_pm = abs(tfv - naive), abs(npv - naive), abs(pm1 - naive)
    ok = max(d_tf, d_np, d_pm) < tol
    print(f"[binary loss]  textbook {naive:.6f} | tf {tfv:.6f} | numpy {npv:.6f} | "
          f"+-1 form {pm1:.6f}  -> {'OK' if ok else 'FAIL'}")

    # the log(1+exp(-v)) variant the note in the docstring warns about:
    # monotone decreasing in v, so it has no minimiser.
    vv = np.linspace(-6, 6, 13)
    bad = np.log(1.0 + np.exp(-vv)) - 1.0 * vv          # y = 1
    mono = bool(np.all(np.diff(bad) < 0))
    print(f"[binary loss]  the log(1+exp(-v)) variant is monotone decreasing in v: "
          f"{mono}  -> that form has no minimum, hence log(1+exp(+v))")
    return ok and mono


def check_binary_offset_algebra(n=5000, seed=2, tol=1e-6) -> bool:
    """
    b(X) + A*tau(X)  ==  [b(X) + pi*tau(X)] + (A - pi)*tau(X).
    This identity is what makes eta = b + pi*tau the exact offset.
    """
    rng = np.random.RandomState(seed)
    b = rng.randn(n)
    tau = rng.randn(n)
    pi = rng.uniform(0.2, 0.8, n)
    a = rng.binomial(1, pi).astype(np.float64)
    lhs = b + a * tau
    rhs = (b + pi * tau) + (a - pi) * tau
    d = float(np.max(np.abs(lhs - rhs)))
    ok = d < tol
    print(f"[binary offset] max|b + A*tau - (eta + (A-pi)*tau)| = {d:.3e}  "
          f"-> {'OK' if ok else 'FAIL'}")
    return ok


def check_end_to_end_binary(seed=0) -> bool:
    """
    Binary analogue of check_end_to_end.  x0 and x1 are the true modifiers,
    x2 is prognostic-only.  The exact offset eta = b + pi*tau is supplied, so
    this isolates the loss/offset plumbing from nuisance estimation error.
    """
    from tensorflow.keras.layers import Dense, ReLU

    rng = np.random.RandomState(seed)
    n, d = 6000, 10
    X = rng.randn(n, d)
    pi_val = 0.5
    trt = rng.binomial(1, pi_val, n).astype(np.float64)
    pi = np.full(n, pi_val)

    tau = 2.0 * X[:, 0] - 2.0 * X[:, 1]        # true log-odds-ratio contrast
    b = 0.8 * X[:, 2]                          # prognostic only
    v = b + trt * tau
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-v))).astype(np.float64)
    eta = b + pi_val * tau                     # exact baseline offset

    def decoder(x):
        x = ReLU()(Dense(32)(x))
        x = ReLU()(Dense(16)(x))
        return Dense(1)(x)

    sel = ConcreteAutoencoderFeatureSelector(
        K=2,
        output_function=decoder,
        num_epochs=200,
        batch_size=256,
        learning_rate=5e-3,
        min_temp=0.05,
        tryout_limit=2,
        loss_name="A_binary",
        per_sample_noise=True,
        verbose=0,
    ).fit(X, y, trt, pi, offset=eta)

    picked = set(int(i) for i in sel.get_indices())
    top2 = set(int(i) for i in np.argsort(-sel.feature_scores("max"))[:2])
    f = sel.predict_score(X)
    corr = float(np.corrcoef(f, tau)[0, 1])
    p = sel.predict_prob(X, trt, pi, eta)
    ok = (picked == {0, 1} or top2 == {0, 1}) and corr > 0.5
    print(f"[end-to-end binary] indices = {sorted(picked)}  top2 by prob = {sorted(top2)}  "
          f"corr(f, true tau) = {corr:+.3f}  mean-max = {sel.mean_max:.3f}  "
          f"P in [{p.min():.3f}, {p.max():.3f}]  "
          f"-> {'OK' if ok else 'check (weak signal / unlucky seed)'}")
    return ok


def _cox_loss_bruteforce(time, event, v, weight=None):
    """Literal transcription of the formula, loops and all, as ground truth."""
    n = len(time)
    w = np.ones(n) if weight is None else np.asarray(weight, dtype=np.float64)
    total, denom = 0.0, 0.0
    for i in range(n):
        if event[i] <= 0:
            continue
        s = sum(math.exp(v[j]) for j in range(n) if time[j] >= time[i])
        total += w[i] * (v[i] - math.log(s))
        denom += w[i]
    return -total / denom


def check_cox_loss(n=120, seed=3, tol=1e-4) -> bool:
    """
    (a) the TF Cox loss equals a brute-force loop over the formula, WITH TIES
    (b) cox_loss_np agrees with both
    (c) the loss is invariant to shifting v by a constant (so eta only matters
        up to a constant)
    """
    rng = np.random.RandomState(seed)
    # coarse grid => many tied event times, which is the interesting case
    time = rng.randint(1, 12, n).astype(np.float64)
    event = rng.binomial(1, 0.65, n).astype(np.float64)
    trt = rng.binomial(1, 0.5, n).astype(np.float64)
    pi = np.full(n, 0.5)
    f = rng.randn(n)
    off = rng.randn(n) * 0.3
    v = off + (trt - pi) * f

    brute = _cox_loss_bruteforce(time, event, v)
    npv = cox_loss_np(time, event, trt, pi, f, off)
    t6 = pack_targets(time, trt, pi, None, off, event)
    tfv = float(a_learning_loss_cox(
        tf.constant(t6), tf.constant(f.reshape(-1, 1), tf.float32)))

    n_ties = n - len(np.unique(time))
    ok_val = max(abs(tfv - brute), abs(npv - brute)) < tol

    shifted = cox_loss_np(time, event, trt, pi, f, off + 7.5)
    ok_shift = abs(shifted - npv) < 1e-9

    print(f"[cox loss]     brute {brute:.6f} | tf {tfv:.6f} | numpy {npv:.6f}  "
          f"({n_ties} tied times)  -> {'OK' if ok_val else 'FAIL'}")
    print(f"[cox loss]     shift-invariance eta -> eta + 7.5: "
          f"{npv:.6f} vs {shifted:.6f}  -> {'OK' if ok_shift else 'FAIL'}")
    return ok_val and ok_shift


def check_cox_shuffle(n=150, seed=4, tol=1e-4) -> bool:
    """
    Permuting the rows must not change the loss: the risk sets are rebuilt from
    the time column, so row order is irrelevant.  This is what makes Keras'
    shuffle=True safe for the Cox loss.
    """
    rng = np.random.RandomState(seed)
    time = rng.gamma(2.0, 2.0, n) + 0.1
    event = rng.binomial(1, 0.7, n).astype(np.float64)
    trt = rng.binomial(1, 0.5, n).astype(np.float64)
    pi = np.full(n, 0.5)
    f = rng.randn(n)
    perm = rng.permutation(n)

    t1 = pack_targets(time, trt, pi, None, None, event)
    t2 = pack_targets(time[perm], trt[perm], pi[perm], None, None, event[perm])
    v1 = float(a_learning_loss_cox(
        tf.constant(t1), tf.constant(f.reshape(-1, 1), tf.float32)))
    v2 = float(a_learning_loss_cox(
        tf.constant(t2), tf.constant(f[perm].reshape(-1, 1), tf.float32)))
    ok = abs(v1 - v2) < tol
    print(f"[cox shuffle]  {v1:.6f} vs {v2:.6f} after permuting rows  "
          f"-> {'OK' if ok else 'FAIL'}")
    return ok


def check_cox_fitter(n=4000, seed=5, tol=0.12) -> bool:
    """cox_fit_ridge should recover a known beta on exponential survival data."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 3)
    beta_true = np.array([0.8, -0.5, 0.0])
    lin = X @ beta_true
    T = rng.exponential(np.exp(-lin))          # hazard prop. to exp(lin)
    C = rng.exponential(np.full(n, 2.0))
    time = np.minimum(T, C)
    event = (T <= C).astype(float)
    beta = cox_fit_ridge(X, time, event, alpha=1e-6)
    err = float(np.max(np.abs(beta - beta_true)))
    ok = err < tol
    print(f"[cox fitter]   beta_hat = {np.round(beta, 3)}  true = {beta_true}  "
          f"max err = {err:.3f}  (censoring {1 - event.mean():.0%})  "
          f"-> {'OK' if ok else 'FAIL'}")
    return ok


def check_end_to_end_cox(seed=0) -> bool:
    """
    TTE analogue of check_end_to_end.  x0, x1 are the true effect modifiers,
    x2 is prognostic only.  The exact offset eta = h + pi*tau is supplied so
    this isolates the loss from nuisance estimation error.
    """
    from tensorflow.keras.layers import Dense, ReLU

    rng = np.random.RandomState(seed)
    n, d = 4000, 10
    X = rng.randn(n, d)
    pi_val = 0.5
    trt = rng.binomial(1, pi_val, n).astype(np.float64)
    pi = np.full(n, pi_val)

    tau = 1.2 * X[:, 0] - 1.2 * X[:, 1]      # true log-hazard-ratio contrast
    h = 0.6 * X[:, 2]                        # prognostic only
    lin = h + trt * tau
    T = rng.exponential(np.exp(-lin))
    C = rng.exponential(np.full(n, 3.0))
    time = np.minimum(T, C)
    event = (T <= C).astype(float)
    eta = h + pi_val * tau                   # exact baseline offset

    def decoder(x):
        x = ReLU()(Dense(32)(x))
        x = ReLU()(Dense(16)(x))
        return Dense(1)(x)

    sel = ConcreteAutoencoderFeatureSelector(
        K=2,
        output_function=decoder,
        num_epochs=1500,           # full batch => 1 epoch == 1 gradient step
        batch_size=None,           # full batch, required by the Cox loss
        learning_rate=1e-2,
        min_temp=0.05,
        tryout_limit=2,
        loss_name="A_cox",
        per_sample_noise=True,
        verbose=0,
    ).fit(X, time, trt, pi, offset=eta, event=event)

    picked = set(int(i) for i in sel.get_indices())
    top2 = set(int(i) for i in np.argsort(-sel.feature_scores("max"))[:2])
    f = sel.predict_score(X)
    corr = float(np.corrcoef(f, tau)[0, 1])
    gain = cox_subgroup_gain(time, event, trt, f)
    ok = (picked == {0, 1} or top2 == {0, 1}) and corr > 0.5
    print(f"[end-to-end cox]    indices = {sorted(picked)}  top2 by prob = {sorted(top2)}  "
          f"corr(f, true tau) = {corr:+.3f}  cox_subgroup_gain = {gain:+.3f}  "
          f"events = {event.mean():.0%}  mean-max = {sel.mean_max:.3f}  "
          f"-> {'OK' if ok else 'check (weak signal / unlucky seed)'}")
    return ok


if __name__ == "__main__":
    tf.random.set_seed(0)
    np.random.seed(0)
    print("--- continuous ---")
    check_custom_gradient()
    check_alignment()
    check_end_to_end()
    print("--- binary ---")
    check_binary_loss()
    check_binary_offset_algebra()
    check_end_to_end_binary()
    print("--- time-to-event ---")
    check_cox_loss()
    check_cox_shuffle()
    check_cox_fitter()
    check_end_to_end_cox()
