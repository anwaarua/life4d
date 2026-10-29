# Life4D: calibrated validity of stored 3D geometry across long observation gaps

Code and evaluation protocol for Life4D: calibrated validity of stored 3D geometry across long observation gaps (A. Ulhaq, 2026).

## Contents
- `simulation/life4d_sim.py` – controlled study (Table 1, cross-site pooling). Runs in about 1 minute with NumPy only:
  `python3 simulation/life4d_sim.py` (expected output: `simulation/expected_output_10_seeds.txt`)
- `simulation/make_preprint_fig.py` – reliability diagram. Needs matplotlib.
- `pilot/life4d_pilot.py` – airborne LiDAR pilot on bi-temporal change benchmarks (Urb3DCD-v2, simulated; AHN-CD, real). See `pilot/README.md`.
- `pilot/make_test_pairs.py` – small synthetic pairs to test the pilot code (not results).

## Data
Urb3DCD-v2 (IEEE DataPort, CC BY 4.0) and AHN-CD (de Gélis et al., IEEE TGRS 62, 2024). Data are not redistributed here.

## Citation
A. Ulhaq, Life4D pilot code and evaluation protocol, github.com/anwaarua/life4d, 2026.

## Licence
MIT (see LICENSE).
