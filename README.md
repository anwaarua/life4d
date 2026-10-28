# Life4D: calibrated validity of stored 3D geometry across long observation gaps

Code for the preprint (folder `life4d/` of github.com/anwaarua/AgenticAI) *"Seen, Exists, Valid: Calibrated Validity of Stored 3D Geometry Across Long Observation Gaps"* (A. Ulhaq, 2026).

## Contents
- `simulation/life4d_sim.py` – controlled study (Table 1, cross-site pooling). Runs in about 1 minute with NumPy only:
  `python3 simulation/life4d_sim.py`
- `simulation/make_preprint_fig.py` – reliability diagram (Figure 1). Needs matplotlib.
- `pilot/life4d_pilot.py` – real airborne LiDAR pilot on bi-temporal benchmarks (Urb3DCD-v2, AHN-CD). See `pilot/README.md`.
- `pilot/make_test_pairs.py` – small synthetic pairs to test the pilot code (not results).

## Data
Urb3DCD-v2 (IEEE DataPort, CC BY 4.0) and AHN-CD (de Gélis et al., arXiv:2304.12639). Data are not redistributed here.

## Citation
A. Ulhaq, "Seen, Exists, Valid: Calibrated Validity of Stored 3D Geometry Across Long Observation Gaps," arXiv, 2026.

## Licence
MIT (see LICENSE).
