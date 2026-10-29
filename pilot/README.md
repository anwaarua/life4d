# Life4D real-data pilot

Tests whether stored geometry from survey A is still valid at survey B when part of B is not observed
(simulated unflown strips and occlusions), on bi-temporal airborne LiDAR change benchmarks.

## Data (free, CC BY 4.0; not redistributed here)
- **Urb3DCD-v2**: simulated airborne LiDAR over Lyon, two epochs per scene, change labels (IEEE DataPort).
  Processing scripts: GitHub `coledea/Multi-Temporal-Point-Cloud-Datasets`.
- **AHN-CD**: real Dutch national airborne LiDAR (AHN3 vs AHN4) with change labels (de Gélis et al., IEEE TGRS 62, 2024).

## Install
```
pip install numpy            # PLY and TXT readers are built in
pip install "laspy[lazrs]"   # only for .las/.laz files
```

## Check file names and label codes
Read the dataset README and note the file name of each epoch, the change-label field on epoch B and the
codes for "unchanged" and "mobile objects". To print the fields of a PLY file:
```
python3 -c "import life4d_pilot as p; d=p.read_ply('SCENE/pc1.ply'); print(list(d), {k: sorted(set(d[k].astype(int)))[:10] for k in d if 'label' in k})"
```

## Run
```
python3 life4d_pilot.py --root /path/to/Urb3DCD-v2 --a-pattern "pc0*" --b-pattern "pc1*" \
   --label-field label_ch --unchanged 0 --transient <mobile-code> --dataset-name "Urb3DCD-v2" --out results
```
To test the code itself first: `python3 make_test_pairs.py /tmp/test 12` and run the same command on
`/tmp/test`. Numbers from these synthetic test pairs are not results.

## Outputs
`results/pilot_results.json` (all metrics, including unobserved cells, re-survey capture and AUROC),
`results/pilot_table.tex` (LaTeX rows), `results/pilot_table_c1.tex` (compact rows) and
`results/pilot_sentence.tex` (one-sentence summary). The settings used for the reported Urb3DCD-v2 run are in
`results_urb3dcd_v2/RUN.md`.

## Notes
Changes are rare in real data, so keeping the last observation can score close to Life4D on overall error; the
expected advantages are calibration and re-survey ranking on unobserved cells. Check the class codes carefully:
a wrong "unchanged" code silently breaks the ground truth.
