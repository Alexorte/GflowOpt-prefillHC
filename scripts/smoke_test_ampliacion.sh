#!/usr/bin/env bash
# Prueba rápida de train_gflownet_v2.py en Asia: cada configuración unas pocas iteraciones.
# Ejecutar en el clúster ANTES de lanzar run_ampliacion.sh (el código JAX de la ampliación no se ha
# ejecutado todavía). Uso:  bash scripts/smoke_test_ampliacion.sh   (RUN="..." igual que en run_ampliacion.sh)
set -u
RUN=${RUN:-python}
ITERS=${ITERS:-300}
OUT=${OUT:-smoke_ampliacion}
export WANDB_MODE=${WANDB_MODE:-offline}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p "$OUT"

COMMON="--batch_size 64 --num_iterations $ITERS --history_training 100 --num_samples_history_posterior 50 \
 --num_samples_posterior 100 --eval_params final --seed 0"
FIX="--update_target_every 50 --logit_scale_init 3.0 --fixed_epsilon 0.1 --max_parents 5"
PREFILL="--prefill 10 --hc_epsilon 0.2 --hc_top_k 5"
WARM="$PREFILL --expert_buffer --backward_K 5 --boundary_M 10 --pretrain_steps 20 --hc_every 100 --hc_top 2 --hc_K 3"

declare -A CFG
CFG[base]="--lr 1e-4 --prefill 0"
CFG[P]="--lr 1e-4 $PREFILL"
CFG[PW]="--lr 1e-4 $WARM"
CFG[I]="--lr 3e-5 --prefill 0 $FIX --informed --beta0 0.1 --beta_end 1.0"
CFG[IW]="--lr 3e-5 $FIX --informed $WARM"

fail=0
for cfg in base P PW I IW; do
  dir="$OUT/asia_$cfg"
  echo "== $cfg"
  if $RUN train_gflownet_v2.py $COMMON ${CFG[$cfg]} --output_folder "$dir" asia_interventional_bic > "$OUT/$cfg.log" 2>&1 \
     && [ -f "$dir/results.json" ]; then
    echo "   OK  ($dir/results.json)"
  else
    echo "   FALLO: ver $OUT/$cfg.log"; tail -n 20 "$OUT/$cfg.log"; fail=1
  fi
done
exit $fail
