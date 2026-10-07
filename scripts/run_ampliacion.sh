#!/usr/bin/env bash
# Comparación de la ampliación: base TFG vs F (ajustes de entrenamiento) vs I (política informada).
# Uso (desde la raíz del repo):  nohup bash scripts/run_ampliacion.sh > logs_ampliacion/run.log 2>&1 &
# Variables opcionales: DATASETS, SEEDS, ITERS, PARALLEL, ENV_NAME
set -u
DATASETS=${DATASETS:-"child_interventional_bic alarm_interventional_bic"}
SEEDS=${SEEDS:-"0 1 2"}
ITERS=${ITERS:-30000}
PARALLEL=${PARALLEL:-3}
ENV_NAME=${ENV_NAME:-gflowOpt}
OUT=${OUT:-resultados_ampliacion}
mkdir -p "$OUT" logs_ampliacion

# Varias ejecuciones comparten la GPU: que JAX no reserve toda la memoria al arrancar
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.30
export WANDB_MODE=offline

COMMON="--batch_size 256 --prefill 0 --num_iterations $ITERS --history_training 2000 \
 --lr_scheduler reduce_on_plateau --lr_patience 100 --lr_factor 0.3 --lr_min 1e-7"
FIX="--update_target_every 50 --logit_scale_init 3.0 --fixed_epsilon 0.1 --max_parents 5"

declare -A CFG
CFG[base]="--lr 1e-4"
CFG[F]="--lr 1e-4 $FIX"
CFG[I]="--lr 3e-5 $FIX --informed"

run_one () {
  local ds=$1 cfg=$2 seed=$3
  local name="${ds%%_*}_${cfg}_s${seed}"
  local dir="$OUT/$name"
  if [ -f "$dir/results.json" ]; then echo "[$(date +%H:%M)] ya existe $name, se salta"; return; fi
  echo "[$(date +%H:%M)] inicio $name"
  micromamba run -n "$ENV_NAME" python train_gflownet_v2.py $COMMON ${CFG[$cfg]} \
      --seed "$seed" --output_folder "$dir" "$ds" > "logs_ampliacion/$name.log" 2>&1
  echo "[$(date +%H:%M)] fin $name (código $?)"
}

for ds in $DATASETS; do
  for seed in $SEEDS; do
    for cfg in base F I; do
      while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do sleep 20; done
      run_one "$ds" "$cfg" "$seed" &
      sleep 5
    done
  done
done
wait
echo "[$(date +%H:%M)] TODO TERMINADO"
