# -*- coding: utf-8 -*-
"""PRAD — Physics-Residual Anomaly Detection.

Self-supervised anomaly detection for CFRP fairing SHM.
All code is self-contained under src/prad/ and does NOT modify
any existing files (models.py, train.py, models_fno.py, etc.).
"""

# ── Feature dimension mapping (34-dim node features) ──
POSITION_DIMS = [0, 1, 2]          # x, y, z
NORMAL_DIMS = [3, 4, 5]            # nx, ny, nz
CURVATURE_DIMS = [6, 7, 8, 9]      # k1, k2, H, K
DISPLACEMENT_DIMS = [10, 11, 12]    # ux, uy, uz
DISP_MAG_DIM = 13                   # |u|
TEMP_DIM = 14                       # temperature
STRESS_DIMS = [15, 16, 17]          # s11, s22, s12
SMISES_DIM = 18                     # von Mises stress
PRINCIPAL_SUM_DIM = 19              # sigma1 + sigma2
THERMAL_SMISES_DIM = 20             # thermal von Mises
STRAIN_DIMS = [21, 22, 23]          # le11, le22, le12
FIBER_DIMS = [24, 25, 26]           # fiber circumferential direction
LAYUP_DIMS = [27, 28, 29, 30]       # layup angles [0/45/-45/90]
CIRCUM_ANGLE_DIM = 31               # circumferential angle
BOUNDARY_FLAG_DIM = 32              # boundary node flag
LOADED_FLAG_DIM = 33                # loaded node flag

# Grouped for anomaly scoring (physics-relevant features)
PHYSICS_DIMS = STRESS_DIMS + [SMISES_DIM, PRINCIPAL_SUM_DIM, THERMAL_SMISES_DIM]
MECHANICAL_DIMS = DISPLACEMENT_DIMS + [DISP_MAG_DIM] + STRAIN_DIMS

IN_CHANNELS = 34
EDGE_ATTR_DIM = 5
