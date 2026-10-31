# Life4D: calibrated validity of stored 3D geometry across long observation gaps

Code and evaluation protocol for Life4D: calibrated validity of stored 3D geometry across long observation gaps (A. Ulhaq, 2026).

## Contents
- `simulation/life4d_sim.py` – controlled study (Table 2 of the project description, cross-site pooling). Runs in about 1 minute with NumPy only:
  `python3 simulation/life4d_sim.py` (expected output: `simulation/expected_output_10_seeds.txt`)
  The output also prints two extra baselines: `CalibratedLast` (shown in Table 2 as "Keep last observation, calibrated") and `GlobalRate` (one shared change rate for all classes; combined error 21.1%, calibration error 0.023), which is omitted from Table 2 for space and still trails the Life4D-style filter (16.5%, 0.017).
- `simulation/make_preprint_fig.py` – reliability diagram. Needs matplotlib.
- `pilot/life4d_pilot.py` – airborne LiDAR pilot on bi-temporal change benchmarks (Urb3DCD-v2, simulated; AHN-CD, real). See `pilot/README.md`.
- `pilot/make_test_pairs.py` – small synthetic pairs to test the pilot code (not results).

## Data
Urb3DCD-v2 (IEEE DataPort, CC BY 4.0) and AHN-CD (de Gélis et al., IEEE TGRS 62, 2024). Data are not redistributed here.

## Citation
A. Ulhaq, Life4D pilot code and protocol, github.com/anwaarua/life4d, 2026.

## Licence
MIT (see LICENSE).
