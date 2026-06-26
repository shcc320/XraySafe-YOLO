# XraySafe-YOLO model configurations

These YAML files are manuscript-aligned Ultralytics model definitions.

- `xraysafe_yolo11n_cgoa.yaml`: YOLO11n + CGOA on final P3/P4 branches; P5 bypasses CGOA.
- `xraysafe_yolo11n_lsaff.yaml`: YOLO11n + three scale-specific LSAFF outputs.
- `xraysafe_yolo11n_full.yaml`: full XraySafe-YOLO = CGOA(P3/P4) + LSAFF(P3'/P4'/P5).

Second-round optimization candidates are kept separate from the formal ablation
configs:

- `xraysafe_yolo11n_opt_resgated.yaml`: CGOA + residual-gated LSAFF on P3/P4/P5.
- `xraysafe_yolo11n_opt_p34_resgated.yaml`: CGOA + residual-gated LSAFF on P3/P4, with P5 bypass.
- `xraysafe_yolo11n_opt_spatial_resgated.yaml`: CGOA + residual-gated LSAFF with spatial scale weights.

Ablation commands use the following model order:

1. `yolo11n.pt`
2. `configs/models/xraysafe_yolo11n_cgoa.yaml`
3. `configs/models/xraysafe_yolo11n_lsaff.yaml`
4. `configs/models/xraysafe_yolo11n_full.yaml`

For the YAML variants, `scripts/run_all.py` attempts partial transfer from
`yolo11n.pt` when `--pretrained yolo11n.pt` is passed.

The optimization script `run_opixray_lsaff_optimization_4070.ps1` uses a separate
output folder and, by default, initializes from the previous CGOA best weights
when available.
