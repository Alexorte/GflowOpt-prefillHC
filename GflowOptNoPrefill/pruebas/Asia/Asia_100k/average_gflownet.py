import json
import re
from pathlib import Path

import numpy as np

# Orden preferido de métricas al imprimir.
# Si aparece alguna más en results.json, también se incluirá.
PREFERRED_METRICS = [
    "expected_shd",
    "expected_edges",
    "roc_auc",
    "prc_auc",
    "ave_prec",
    "median_shd",
    "shd_p10",
    "shd_p90",
    "edge_entropy",
    "bic_mean",
    "bic_max",
    "bdeu_mean",
    "bdeu_max",
    "neg_avg_predictive_log_likelihood",
    "predictive_log_likelihood_best",
    "Prefill_mean_final_edges",
    "Prefill_mean_final_score",
    "Prefill_mean_shd",
    "Prefill_best_shd",
    "Prefill_bic_mean",
    "Prefill_bic_best",
    "Prefill_bdeu_mean",
    "Prefill_bdeu_best",
    "prefill_time_seconds",
    "hc_prefill_time_seconds",
    "training_time_seconds",
    "evaluation_time_seconds",
    "total_time_seconds",
    "time_per_iteration_seconds",
]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def extract_method_and_seed(results_path: Path):
    """
    Espera rutas tipo:
      output_hc_prefill_0/results.json
      output_baseline_3/results.json
    Devuelve:
      method='output_hc_prefill', seed=0
    """
    folder_name = results_path.parent.name
    match = re.match(r"(.+)_([0-9]+)$", folder_name)
    if match:
        method = match.group(1)
        seed = int(match.group(2))
    else:
        method = folder_name
        seed = None
    return method, seed


def summarize_dicts(dicts):
    """
    Calcula media y desviación típica muestral (ddof=1) para todas
    las métricas numéricas presentes en los json.
    """
    all_metrics = set()
    for d in dicts:
        for k, v in d.items():
            if is_number(v):
                all_metrics.add(k)

    ordered_metrics = [m for m in PREFERRED_METRICS if m in all_metrics]
    ordered_metrics += sorted(all_metrics - set(ordered_metrics))

    summary = {}
    for metric in ordered_metrics:
        values = [d[metric] for d in dicts if metric in d and is_number(d[metric])]
        if not values:
            continue
        summary[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "n": len(values),
        }
    return summary


def print_summary(title, summary):
    print(f"\n=== {title} ===")
    for metric, stats in summary.items():
        print(f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f}  (n={stats['n']})")


def main(root_folder="."):
    root = Path(root_folder)

    # Busca todos los results.json de forma recursiva
    results_files = sorted(root.rglob("results.json"))

    if not results_files:
        print("[ERROR] No se ha encontrado ningún results.json")
        return

    grouped_results = {}
    grouped_seeds = {}

    for results_path in results_files:
        method, seed = extract_method_and_seed(results_path)
        data = load_json(results_path)

        grouped_results.setdefault(method, []).append(data)
        if seed is not None:
            grouped_seeds.setdefault(method, []).append(seed)

    output_summary = {}

    for method in sorted(grouped_results.keys()):
        summaries = summarize_dicts(grouped_results[method])

        # Aviso por si no hay exactamente 5 semillas
        seeds = sorted(grouped_seeds.get(method, []))
        if seeds:
            print(f"\n[INFO] {method}: semillas encontradas -> {seeds}")
            if len(seeds) != 5:
                print(f"[WARN] {method}: se esperaban 5 semillas, pero se encontraron {len(seeds)}")

        print_summary(method.upper(), summaries)

        output_summary[method.upper()] = {
            metric: f"{stats['mean']:.4f} ± {stats['std']:.4f}"
            for metric, stats in summaries.items()
        }

    output_path = root / "summary_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_summary, f, indent=4, ensure_ascii=False)

    print(f"\n[OK] Resumen guardado en: {output_path}")


if __name__ == "__main__":
    main(".")