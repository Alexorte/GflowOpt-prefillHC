#!/usr/bin/env bash
# Comparación de la ampliación (train_gflownet_v2.py, solo fase GFlowNet). Escalera de ablación:
#   base : GFlowNet tal cual viene (TFG sin prefill), evaluando los parámetros finales
#   M    : base + máximo 5 padres (referencia justa: mismo espacio de búsqueda que el resto)
#   F    : M + ajustes de entrenamiento (eps fijo 0.1, red objetivo cada 50, escala del stop)
#   LS   : F + búsqueda local guiada por la política -> buffer protegido con muestreo mixto
#   LST  : LS + templado de la recompensa (beta 0.1 -> 1 en la primera mitad)
#   I    : F + política informada por el delta-score
# Opcionales (no se lanzan por defecto; añadir a CONFIGS): P, PW (warm-start del TFG), ILS (I + LS).
# Uso (desde la raíz del repo):  nohup bash scripts/run_ampliacion.sh > logs_ampliacion/run.log 2>&1 &
# Variables opcionales: DATASETS, SEEDS, CONFIGS, ITERS, PARALLEL, RUN, OUT
#   RUN: cómo se lanza Python. Por defecto "python" (entorno ya activado);
#        p.ej. RUN="micromamba run -n gflowOpt python" o RUN="conda run -n gflowOpt python".
set -u
DATASETS=${DATASETS:-"asia_interventional_bic sachs_interventional_bic child_interventional_bic alarm_interventional_bic"}
SEEDS=${SEEDS:-"0 1 2 3 4"}
CONFIGS=${CONFIGS:-"base M F LS LST I"}
ITERS=${ITERS:-30000}
PARALLEL=${PARALLEL:-3}
RUN=${RUN:-python}
OUT=${OUT:-resultados_ampliacion}
mkdir -p "$OUT" logs_ampliacion

# Varias ejecuciones comparten la GPU: que JAX no reserve toda la memoria al arrancar
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.30}
export WANDB_MODE=${WANDB_MODE:-offline}

COMMON="--batch_size 256 --num_iterations $ITERS --history_training 2000 --eval_params final \
 --lr_scheduler reduce_on_plateau --lr_patience 100 --lr_factor 0.3 --lr_min 1e-7"
FIX="--update_target_every 50 --logit_scale_init 3.0 --fixed_epsilon 0.1 --max_parents 5"
PREFILL="--prefill 100 --hc_epsilon 0.2 --hc_top_k 5"
WARM="$PREFILL --expert_buffer --rho_start 0.5 --rho_end 0.1 --backward_K 10 --boundary_M 20"

MP="--max_parents 5"
LS="--expert_buffer --rho_start 0.5 --rho_end 0.2 --ls_every 20 --ls_batch 16 --ls_rounds 4 --ls_destroy 0.5"
TEMP="--beta0 0.1 --beta_end 1.0 --anneal_frac 0.5"

declare -A CFG
CFG[base]="--lr 1e-4 --prefill 0"
CFG[M]="--lr 1e-4 --prefill 0 $MP"
CFG[F]="--lr 1e-4 --prefill 0 $FIX"
CFG[LS]="--lr 1e-4 --prefill 0 $FIX $LS"
CFG[LST]="--lr 1e-4 --prefill 0 $FIX $LS $TEMP"
CFG[I]="--lr 3e-5 --prefill 0 $FIX --informed"
CFG[ILS]="--lr 3e-5 --prefill 0 $FIX --informed $LS"
CFG[P]="--lr 1e-4 $PREFILL"
CFG[PW]="--lr 1e-4 $WARM"

run_one () {
  local ds=$1 cfg=$2 seed=$3
  local name="${ds%%_*}_${cfg}_s${seed}"
  local dir="$OUT/$name"
  if [ -f "$dir/results.json" ]; then echo "[$(date +%H:%M)] ya existe $name, se salta"; return; fi
  echo "[$(date +%H:%M)] inicio $name"
  $RUN train_gflownet_v2.py $COMMON ${CFG[$cfg]} \
      --seed "$seed" --output_folder "$dir" "$ds" > "logs_ampliacion/$name.log" 2>&1
  echo "[$(date +%H:%M)] fin $name (código $?)"
}

for ds in $DATASETS; do
  for seed in $SEEDS; do
    for cfg in $CONFIGS; do
      while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do sleep 20; done
      run_one "$ds" "$cfg" "$seed" &
      sleep 5
    done
  done
done
wait
echo "[$(date +%H:%M)] TODO TERMINADO"
