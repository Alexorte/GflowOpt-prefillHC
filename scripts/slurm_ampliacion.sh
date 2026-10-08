#!/usr/bin/env bash
# Plantilla SLURM (job array): una ejecución de train_gflownet_v2.py por tarea.
# AJUSTAR a la configuración del clúster: partición, GPU, tiempo y activación del entorno.
# Uso:  sbatch --array=0-119%6 scripts/slurm_ampliacion.sh
#   (4 datasets x 6 configuraciones x 5 semillas = 120 tareas; %6 = máximo 6 a la vez)
#SBATCH --job-name=gfn_ampl
#SBATCH --partition=CAMBIAR
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=logs_ampliacion/slurm_%A_%a.out
set -u

# Activación del entorno (ajustar): p.ej. module load ...; conda activate gflowOpt
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate gflowOpt

DATASETS=(asia_interventional_bic sachs_interventional_bic child_interventional_bic alarm_interventional_bic)
CONFIGS=(base P PW F I IW)
SEEDS=(0 1 2 3 4)
ITERS=${ITERS:-30000}
OUT=${OUT:-resultados_ampliacion}

i=${SLURM_ARRAY_TASK_ID}
nc=${#CONFIGS[@]}; ns=${#SEEDS[@]}
ds=${DATASETS[$(( i / (nc * ns) ))]}
cfg=${CONFIGS[$(( (i / ns) % nc ))]}
seed=${SEEDS[$(( i % ns ))]}

export WANDB_MODE=offline
mkdir -p "$OUT" logs_ampliacion
# Reutiliza las configuraciones de run_ampliacion.sh lanzando una sola combinación
DATASETS="$ds" CONFIGS="$cfg" SEEDS="$seed" ITERS="$ITERS" OUT="$OUT" PARALLEL=1 \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 bash scripts/run_ampliacion.sh
