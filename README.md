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
