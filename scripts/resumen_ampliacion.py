"""Tabla resumen de resultados_ampliacion/*/results.json (media ± desviación por red y configuración).
Uso: micromamba run -n gflowOpt python scripts/resumen_ampliacion.py [carpeta]"""
import json, sys, glob, os
import numpy as np

base = sys.argv[1] if len(sys.argv) > 1 else 'resultados_ampliacion'
ORDER = ['base', 'P', 'PW', 'F', 'I', 'IW']
rows = {}
for f in sorted(glob.glob(os.path.join(base, '*', 'results.json'))):
    name = os.path.basename(os.path.dirname(f))          # p.ej. child_I_s0
    net, cfg, _ = name.rsplit('_', 2)
    r = json.load(open(f))
    rows.setdefault((net, cfg), []).append(r)

keys = [('expected_shd', 'E-SHD'), ('expected_edges', 'Aristas'), ('roc_auc', 'AUROC'),
        ('bic_mean', 'BIC medio'), ('bic_max', 'BIC max'), ('training_time_seconds', 'Tiempo (s)')]
print(f"{'red':8s} {'config':6s} {'n':>2s} " + ' '.join(f'{h:>20s}' for _, h in keys))
for (net, cfg), rs in sorted(rows.items(), key=lambda x: (x[0][0], ORDER.index(x[0][1]) if x[0][1] in ORDER else 9)):
    cells = []
    for k, _ in keys:
        v = [r[k] for r in rs if k in r and isinstance(r[k], (int, float))]
        cells.append(f'{np.mean(v):10.2f} ± {np.std(v):7.2f}' if v else f'{"-":>20s}')
    print(f'{net:8s} {cfg:6s} {len(rs):2d} ' + ' '.join(cells))
