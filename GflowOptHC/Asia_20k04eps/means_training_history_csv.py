import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


# Orden preferido de métricas al imprimir y guardar.
# Si aparece alguna métrica adicional en el CSV, también se incluirá al final.
PREFERRED_METRICS = [
    "training_time_seconds",
    "history_eval_time_seconds",
    "expected_shd",
    "expected_edges",
    "roc_auc",
    "prc_auc",
    "ave_prec",
    "bic_mean",
    "bic_max",
    "bdeu_mean",
    "bdeu_max",
    "loss",
    "avg_loss",
    "smooth_loss",
    "epsilon",
    "learning_rate",
    "neg_avg_predictive_log_likelihood",
    "predictive_log_likelihood_best",
]


def extract_method_and_seed(csv_path: Path):
    """
    Espera rutas tipo:
      output_hc_prefill_0/training_history_gflownet.csv
      output_baseline_3/training_history_gflownet.csv

    Devuelve:
      method='output_hc_prefill', seed=0
    """
    folder_name = csv_path.parent.name
    match = re.match(r"(.+)_([0-9]+)$", folder_name)
    if match:
        method = match.group(1)
        seed = int(match.group(2))
    else:
        method = folder_name
        seed = None
    return method, seed


def load_training_history(csv_path: Path, seed: int | None):
    df = pd.read_csv(csv_path)

    if "iteration" not in df.columns:
        raise ValueError(f"El archivo no tiene columna 'iteration': {csv_path}")

    df["iteration"] = pd.to_numeric(df["iteration"], errors="raise").astype(int)

    if seed is not None:
        df["seed"] = seed

    return df


def numeric_metric_columns(df: pd.DataFrame):
    ignored = {"iteration", "seed"}
    metrics = []

    for col in df.columns:
        if col in ignored:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            metrics.append(col)

    ordered = [m for m in PREFERRED_METRICS if m in metrics]
    ordered += sorted(set(metrics) - set(ordered))
    return ordered


def summarize_history(dfs: list[pd.DataFrame]):
    """
    Calcula media y desviación típica muestral por iteración.

    Importante:
      - La desviación se calcula con ddof=1, como en el script de results.json.
      - Los NaN se ignoran por métrica e iteración.
      - Si una métrica solo existe en una semilla/iteración, std=0.0.
      - No se guardan columnas _n en el CSV final para mantenerlo limpio.
    """
    all_df = pd.concat(dfs, ignore_index=True, sort=False)
    metrics = numeric_metric_columns(all_df)

    rows = []
    for iteration, group in all_df.groupby("iteration", sort=True):
        row = {"iteration": int(iteration)}

        for metric in metrics:
            values = group[metric].dropna().to_numpy(dtype=float)
            n = int(len(values))

            if n == 0:
                row[f"{metric}_mean"] = np.nan
                row[f"{metric}_std"] = np.nan
            else:
                row[f"{metric}_mean"] = float(np.mean(values))
                row[f"{metric}_std"] = float(np.std(values, ddof=1)) if n > 1 else 0.0

        rows.append(row)

    return pd.DataFrame(rows).sort_values("iteration").reset_index(drop=True), metrics


def make_formatted_summary(summary_df: pd.DataFrame, metrics: list[str], decimals: int):
    """
    Genera una versión cómoda para leer en tablas:
      iteration | expected_shd | roc_auc | ...
    donde cada celda tiene formato 'media ± desviación'.
    """
    formatted = pd.DataFrame({"iteration": summary_df["iteration"]})

    for metric in metrics:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"

        def fmt(row):
            if pd.isna(row[mean_col]):
                return ""
            return f"{row[mean_col]:.{decimals}f} ± {row[std_col]:.{decimals}f}"

        formatted[metric] = summary_df.apply(fmt, axis=1)

    return formatted


def print_last_iteration_summary(method: str, summary_df: pd.DataFrame, metrics: list[str], decimals: int):
    last = summary_df.iloc[-1]
    last_iteration = int(last["iteration"])

    print(f"\n=== {method.upper()} | última iteración: {last_iteration} ===")
    for metric in metrics:
        mean = last[f"{metric}_mean"]
        std = last[f"{metric}_std"]

        if pd.isna(mean):
            continue
        print(f"{metric}: {mean:.{decimals}f} ± {std:.{decimals}f}")


def warn_missing_iterations(method: str, dfs: list[pd.DataFrame], seeds: list[int | None]):
    iterations_by_seed = {}

    for df, seed in zip(dfs, seeds):
        label = f"seed_{seed}" if seed is not None else "seed_desconocida"
        iterations_by_seed[label] = set(df["iteration"].astype(int).tolist())

    if len(iterations_by_seed) <= 1:
        return

    all_iterations = set.union(*iterations_by_seed.values())
    for label, its in iterations_by_seed.items():
        missing = sorted(all_iterations - its)
        if missing:
            preview = missing[:10]
            suffix = "..." if len(missing) > 10 else ""
            print(f"[WARN] {method}: {label} no tiene {len(missing)} iteraciones: {preview}{suffix}")


def main(root_folder=".", file_name="training_history_gflownet.csv", decimals=4):
    root = Path(root_folder)
    csv_files = sorted(root.rglob(file_name))

    if not csv_files:
        print(f"[ERROR] No se ha encontrado ningún archivo llamado {file_name}")
        return

    grouped_dfs = {}
    grouped_seeds = {}
    grouped_paths = {}

    for csv_path in csv_files:
        method, seed = extract_method_and_seed(csv_path)
        df = load_training_history(csv_path, seed)

        grouped_dfs.setdefault(method, []).append(df)
        grouped_seeds.setdefault(method, []).append(seed)
        grouped_paths.setdefault(method, []).append(csv_path)

    for method in sorted(grouped_dfs.keys()):
        dfs = grouped_dfs[method]
        seeds = grouped_seeds[method]
        paths = grouped_paths[method]

        detected_seeds = sorted(seed for seed in seeds if seed is not None)
        print(f"\n[INFO] {method}: archivos encontrados -> {len(paths)}")

        if detected_seeds:
            print(f"[INFO] {method}: semillas encontradas -> {detected_seeds}")
            if len(detected_seeds) != 5:
                print(f"[WARN] {method}: se esperaban 5 semillas, pero se encontraron {len(detected_seeds)}")
        else:
            print(f"[WARN] {method}: no se ha podido extraer la semilla del nombre de las carpetas")

        warn_missing_iterations(method, dfs, seeds)

        summary_df, metrics = summarize_history(dfs)
        formatted_df = make_formatted_summary(summary_df, metrics, decimals)

        summary_path = root / f"summary_training_history_{method}.csv"
        formatted_path = root / f"summary_training_history_{method}_formatted.csv"

        summary_df.to_csv(summary_path, index=False)
        formatted_df.to_csv(formatted_path, index=False)

        print_last_iteration_summary(method, summary_df, metrics, decimals)
        print(f"[OK] CSV numérico guardado en: {summary_path}")
        print(f"[OK] CSV formateado guardado en: {formatted_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calcula media ± desviación de training_history_gflownet.csv por iteración y por método."
    )
    parser.add_argument("root_folder", nargs="?", default=".", help="Carpeta raíz donde buscar los CSV")
    parser.add_argument("--file_name", default="training_history_gflownet.csv", help="Nombre del CSV a buscar")
    parser.add_argument("--decimals", type=int, default=4, help="Número de decimales al formatear resultados")
    args = parser.parse_args()

    main(args.root_folder, args.file_name, args.decimals)
