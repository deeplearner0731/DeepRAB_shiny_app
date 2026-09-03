#!/bin/bash
cd "C:/Users/liuzx18/OneDrive - AbbVie Inc (O365)/Desktop/code_folder/Deep_neural_network_subgroup"
export TF_USE_LEGACY_KERAS=1
export TF_CPP_MIN_LOG_LEVEL=3
export OMP_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1
for r in $(seq 0 9); do
  /c/dnnenv/Scripts/python.exe run_synthetic_worker.py --rep "$r" > "logs/rep$r.log" 2>&1 &
done
wait
echo "ALL REPS FINISHED"
