# GFlowOpt-prefillHC: aprendizaje de estructuras de Redes Bayesianas con prefill dirigido del replay buffer

![Arquitectura de GFlowOpt](modelfig3_00.png)

Este repositorio contiene el código desarrollado para el Trabajo Fin de Grado:

**“Optimización y Aceleración del Aprendizaje de Estructuras de Redes Bayesianas mediante Redes de Flujo Generativas con Muestreo Sesgado”**

El proyecto parte del código original de **DAG-GFlowNet / GFlowOpt** para el aprendizaje de estructuras de Redes Bayesianas. El marco original combina tres fases principales:

1. entrenamiento de una GFlowNet para muestrear estructuras de Redes Bayesianas;
2. entrenamiento de un modelo proxy a partir de las estructuras generadas por la GFlowNet;
3. aplicación de una fase final de optimización/refinamiento para obtener DAGs de alta puntuación.

Este repositorio extiende dicho marco incorporando una estrategia de **arranque en caliente** (*warm-start*) basada en un **prefill dirigido mediante Hill Climbing**. La idea principal es inicializar el replay buffer de la GFlowNet con trayectorias asociadas a estructuras prometedoras antes de comenzar la fase principal de entrenamiento. Con ello, se busca proporcionar una señal inicial más informativa y comparar el efecto de esta inicialización frente a configuraciones con prefill aleatorio y sin prefill.

## Modificaciones principales

Las principales modificaciones introducidas en este repositorio son:

* Implementación de una estrategia de prefill dirigido del replay buffer basada en Hill Climbing.
* Implementación de una variante de prefill aleatorio para comparación experimental.
* Soporte para entrenamiento sin prefill del replay buffer.
* Adaptación de los scripts de entrenamiento de la GFlowNet a las distintas configuraciones experimentales.
* Incorporación de métricas de evaluación como BIC, BDeu y métricas predictivas auxiliares.
* Incorporación de mecanismos de caché para reducir evaluaciones repetidas de puntuación durante el prefill dirigido.
* Organización de scripts y comandos para reproducir los experimentos realizados en el TFG.

## Estructura del repositorio

```text
.
├── dag_gflownet/                    # Módulos principales de DAG-GFlowNet
├── data/                            # Conjuntos de datos benchmark
├── HC.py                            # Utilidades de Hill Climbing y optimización final
├── latent_optimize.py               # Optimización continua en el espacio latente
├── optimize_gflownet.py             # Fase final de optimización
├── proxy_model.py                   # Definición del modelo proxy
├── train_gflownet_prefill_hc.py     # Entrenamiento de GFlowNet con prefill Hill Climbing
├── train_gflownet_prefill_random.py # Entrenamiento de GFlowNet con prefill aleatorio
├── train_proxy_from_gflownet.py     # Entrenamiento del modelo proxy
├── requirements.txt                 # Dependencias principales de Python
└── jsp1.yml                         # Configuración del entorno
```

## Instalación

Clonar el repositorio e instalar las dependencias principales:

```bash
pip install -r requirements.txt
```

De forma alternativa, se proporciona el archivo de configuración del entorno utilizado durante el desarrollo:

```text
jsp1.yml
```

En caso de ejecutar los experimentos en GPU, puede ser necesario adaptar la instalación de JAX/CUDA a la versión de CUDA y al driver de NVIDIA disponibles en la máquina.

## Flujo de trabajo

El proceso completo para obtener estructuras de Redes Bayesianas de alta puntuación consta de tres fases principales:

1. entrenar el modelo GFlowNet;
2. entrenar el modelo proxy usando estructuras muestreadas por la GFlowNet entrenada;
3. aplicar la fase final de optimización para obtener DAGs refinados.

El repositorio permite ejecutar tres configuraciones de entrenamiento de la GFlowNet:

* GFlowNet con prefill dirigido mediante Hill Climbing;
* GFlowNet con prefill aleatorio;
* GFlowNet sin prefill.

## 1. Entrenamiento de GFlowNet con prefill Hill Climbing

Ejemplo para el conjunto de datos Asia:

```bash
python train_gflownet_prefill_hc.py \
  --lr 1e-4 \
  --lr_scheduler reduce_on_plateau \
  --lr_patience 100 \
  --lr_factor 0.3 \
  --lr_min 1e-7 \
  --batch_size 256 \
  --prefill 100 \
  --hc_epsilon 0.2 \
  --hc_top_k 5 \
  --seed 0 \
  --history_every 10 \
  --history_training 1000 \
  --output_folder output_hc_prefill_0 \
  asia_interventional_bic
```

## 2. Entrenamiento de GFlowNet con prefill aleatorio

```bash
python train_gflownet_prefill_random.py \
  --lr 1e-4 \
  --lr_scheduler reduce_on_plateau \
  --lr_patience 100 \
  --lr_factor 0.3 \
  --lr_min 1e-7 \
  --batch_size 256 \
  --prefill 100 \
  --seed 0 \
  --history_training 1000 \
  --output_folder output_random_prefill_0 \
  asia_interventional_bic
```

## 3. Entrenamiento de GFlowNet sin prefill

La configuración sin prefill puede ejecutarse fijando:

```bash
--prefill 0
```

Por ejemplo:

```bash
python train_gflownet_prefill_hc.py \
  --lr 1e-4 \
  --lr_scheduler reduce_on_plateau \
  --lr_patience 100 \
  --lr_factor 0.3 \
  --lr_min 1e-7 \
  --batch_size 256 \
  --prefill 0 \
  --seed 0 \
  --history_training 500 \
  --output_folder output_no_prefill_0 \
  asia_interventional_bic
```

## 4. Entrenamiento del modelo proxy

Una vez entrenado el modelo GFlowNet, se puede entrenar el modelo proxy a partir de las estructuras generadas:

```bash
python train_proxy_from_gflownet.py train \
  --gflownet_model_path output_hc_prefill_0/model.npz \
  --output_dir output_hc_prefill_0/proxy \
  --num_samples 5000 \
  --num_epochs 800 \
  --batch_size 256 \
  --proxy_lr 1e-3 \
  --normalization_method standard \
  --norm_scale_factor 1 \
  --lr_patience 100 \
  --lr_factor 0.3 \
  --lr_threshold 0.0001 \
  --lr_min 1e-7 \
  asia_interventional_bic
```

## 5. Optimización final

La fase final de optimización utiliza la GFlowNet entrenada y el modelo proxy para identificar y refinar DAGs candidatos de alta puntuación.

```bash
python HC.py optimize \
  --gflownet_model_path output_hc_prefill_seed0/model.npz \
  --proxy_model_path output_hc_prefill_seed0/proxy/proxy_model.pkl \
  --output_dir output_hc_prefill_seed0/final_optimization \
  asia_interventional_bic
```

## Conjuntos de datos disponibles

El framework soporta los siguientes conjuntos de datos benchmark:

* `asia_interventional_bic`
* `sachs_interventional_bic`
* `child_interventional_bic`
* `alarm_interventional_bic`
* `hailfinder_interventional_bic`
* `win95pts_interventional_bic`

También pueden añadirse conjuntos de datos personalizados dependiendo de la configuración local.

## Comparación experimental

La comparación experimental principal estudiada en el TFG se realiza entre:

* GFlowNet sin prefill;
* GFlowNet con prefill aleatorio;
* GFlowNet con prefill dirigido mediante Hill Climbing.

Las principales métricas de evaluación utilizadas son:

* Structural Hamming Distance (SHD);
* número de aristas;
* puntuación BIC;
* puntuación BDeu;
* tiempo de entrenamiento;
* métricas predictivas auxiliares.

## Notas

Este repositorio tiene como objetivo reproducir los experimentos realizados en el TFG y documentar las modificaciones introducidas sobre el marco original GFlowOpt. Algunos comandos pueden requerir pequeñas adaptaciones dependiendo del conjunto de datos, el directorio de salida y el entorno computacional utilizado.

## Ampliación (rama `ampliacion-politica-informada`)

### Correcciones respecto al TFG

* **Orden de nodos del ground truth.** En Alarm (y en cualquier red cuyo orden de pgmpy no coincida con el `fixed_order` de los datos) el SHD/AUROC se calculaba contra una red permutada: por eso el AUROC de Alarm salía ≈ 0,50 en todas las configuraciones. Ahora se usa `nodelist=list(data.columns)` en todos los scripts (líneas marcadas con `# FIX orden de nodos (Alarm)`). Solo afecta a las métricas frente al grafo real, no al entrenamiento ni al BIC. **Las métricas de Alarm (y probablemente Hailfinder) del TFG deben recalcularse.**
* **Modelo evaluado.** Los scripts del TFG sustituyen al final los parámetros por los de *menor pérdida media* (ventana de 200 iteraciones) y ese es el `model.npz` que usan las fases 2 y 3. En una GFlowNet off-policy la pérdida depende del contenido del buffer, de modo que el prefill cambia qué checkpoint se elige y las configuraciones no son comparables. En `train_gflownet_v2.py` se evalúan por defecto los parámetros finales (`--eval_params final`); `--eval_params best_loss` reproduce el TFG. Se guardan ambos (`model_final.npz`, `model_best_loss.npz`; `model.npz` es el evaluado).
* **Red base exacta.** Sin `--informed`, la red de `train_gflownet_v2.py` es ahora exactamente la original (antes se le pasaba una matriz de delta-scores de ceros que restaba log(nº acciones válidas) al logit de parada).

### Problemas identificados en el warm-start del TFG

1. **Dilución.** 100 episodios HC son ~10²–10³ transiciones en un buffer FIFO de 100k que recibe 8 transiciones por iteración y que al principio se llena con exploración uniforme: tras unos cientos de iteraciones las transiciones HC son minoría y hacia la iteración ~12k se han sobrescrito. → buffer experto protegido con muestreo mixto (`--expert_buffer`, `--rho_start`, `--rho_end`).
2. **Sesgo de parada.** Todas las transiciones HC tienen Δscore > 0 y no hay ejemplos cerca del óptimo donde añadir empeore: la red aprende que añadir aristas siempre compensa y baja P(stop) → grafos más densos. → transiciones frontera G* → G*+arista (`--boundary_M`) y trayectorias en órdenes aleatorios (`--backward_K`).

Diagnósticos añadidos a `training_history.csv`: `heuristic_fraction_batch` (fracción esperada de transiciones heurísticas por minilote), `buffer_main_size`, `buffer_expert_size`, `expert_pstop_mean` y `expert_log_pstop_mean` (P(stop) media en los DAGs del prefill).

### Opciones de `train_gflownet_v2.py`

| Opción | Qué hace |
| --- | --- |
| `--eval_params final\|best_loss` | Parámetros evaluados/guardados (por defecto `final`) |
| `--informed` | Política informada: logits de cada arista = beta·ΔBIC + salida de la red |
| `--logit_scale_init 3.0` | Escala aprendible del logit de parada (3.0 ≈ ×20) |
| `--update_target_every 50` | Red objetivo más frecuente (ya existía; por defecto 1000) |
| `--fixed_epsilon 0.1` | Exploración fija del 10 % (ya existía; por defecto la exploración empieza 100 % uniforme) |
| `--max_parents 5` | Máximo de padres por nodo |
| `--beta0`, `--beta_end`, `--anneal_frac` | Templado de la recompensa: log R = beta·BIC |
| `--expert_buffer`, `--rho_start`, `--rho_end` | Buffer experto protegido y muestreo mixto |
| `--backward_K`, `--boundary_M` | Trayectorias hacia atrás y transiciones frontera desde los DAGs del prefill |
| `--pretrain_steps` | Preentrenamiento sobre el buffer experto |
| `--hc_every`, `--hc_top`, `--hc_K` | Hill Climbing dentro del bucle de entrenamiento |
| `--ls_every`, `--ls_batch`, `--ls_rounds`, `--ls_destroy` | Búsqueda local guiada por la política (ver abajo) |

### Búsqueda local guiada por la política (`--ls_every`)

Inspirada en *Local Search GFlowNets* (Kim et al., ICLR 2024). Cada `ls_every` iteraciones se toman los `ls_batch` mejores DAGs muestreados recientemente por la política y se aplican `ls_rounds` rondas de: quitar al azar una fracción `ls_destroy` de las aristas, reconstruir el DAG con la política actual y aceptar si el BIC mejora. Las trayectorias aceptadas se insertan en el buffer (con `--expert_buffer`, en el protegido, del que sale una fracción `rho` de cada minilote). A diferencia del prefill HC, los datos de alta recompensa se generan durante todo el entrenamiento y con la propia política, así que no se diluyen. La pérdida DB es off-policy, por lo que siguen siendo datos válidos. Diagnósticos en `training_history.csv`: `ls_accept_rate`, `ls_mean_gain`, `ls_best_score`, `ls_num_transitions`. Código: `dag_gflownet/local_search.py`.

Módulos nuevos: `dag_gflownet/local_search.py`, `dag_gflownet/gflownet_informed.py`, `dag_gflownet/nets/gflownet_informed.py`, `dag_gflownet/warmstart_v2.py`, `dag_gflownet/utils/replay_buffer_mixed.py`.

### Ejecución en el clúster

El código JAX de la ampliación **todavía no se ha ejecutado**: primero, la prueba rápida (6 configuraciones × 300 iteraciones en Asia, cubre todas las opciones nuevas):

```bash
RUN="python" bash scripts/smoke_test_ampliacion.sh
```

`RUN` indica cómo lanzar Python (con el entorno ya activado basta `python`; si no, p. ej. `RUN="conda run -n gflowOpt python"`).

Comparación completa (base, M, F, LS, LST, I; 4 redes × 5 semillas), en segundo plano:

```bash
nohup bash scripts/run_ampliacion.sh > logs_ampliacion/run.log 2>&1 &
```

o, si el clúster usa SLURM, ajustando partición y entorno en `scripts/slurm_ampliacion.sh`:

```bash
sbatch --array=0-119%6 scripts/slurm_ampliacion.sh
```

Resumen de resultados (media ± desviación por red y configuración):

```bash
python scripts/resumen_ampliacion.py resultados_ampliacion
```

Escalera de ablación (cada paso añade una sola cosa, para poder atribuir cada mejora):

| Config | Qué es |
| --- | --- |
| `base` | GFlowNet tal cual viene (TFG sin prefill), evaluando los parámetros finales |
| `M` | base + máximo 5 padres (referencia justa: mismo espacio de búsqueda que las demás) |
| `F` | M + ajustes de entrenamiento (ε fijo 0,1, red objetivo cada 50, escala del stop) |
| `LS` | F + búsqueda local guiada por la política → buffer protegido |
| `LST` | LS + templado de la recompensa (β 0,1 → 1 en la primera mitad) |
| `I` | F + política informada por el delta-score |

Opcionales (añadir a `CONFIGS`): `ILS` (I + LS), `P` y `PW` (warm-start del TFG). LS e I hacen más cálculo por iteración: hay que comparar también por tiempo (`training_time_seconds` en `training_history.csv`), no solo por iteraciones.

Notas:

* Las fases 2 y 3 (`train_proxy_from_gflownet.py`, `HC.py`) crean su propio `DAGGFlowNet`. Un modelo entrenado con `--informed` hay que cargarlo con `InformedDAGGFlowNet` (todavía no está adaptado); los demás son compatibles.
* `wandb` se usa en modo offline en los scripts (`WANDB_MODE=offline`).
