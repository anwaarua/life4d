# Life4D real-data pilot: what to run, step by step

Goal: one real-LiDAR row in C1 Table 3 and the preprint, and one sentence on re-survey efficiency.
Time: about 1–2 days of your effort plus download time. Runs on a laptop (16 GB RAM) or CQU HPC.

## 1. Get the data (both are free; CC BY 4.0)
- **Urb3DCD-v2**: simulated airborne LiDAR, 30 scenes × 2 epochs, PLY, 7 change classes. Hosted on IEEE DataPort (free account).
  Processing scripts: GitHub `coledea/Multi-Temporal-Point-Cloud-Datasets`.
- **AHN-CD**: real Dutch national airborne LiDAR (AHN3 vs AHN4) with change labels (de Gélis et al., arXiv:2304.12639).
  Check that paper's data link.
Use Urb3DCD-v2 first (more scenes). Add AHN-CD if time allows (it is the "real data" headline).

## 2. Install
```
pip install numpy            # PLY and TXT readers are built in
pip install "laspy[lazrs]"   # only if your files are .las/.laz
```

## 3. Check file names and label codes
Open the dataset README. Note:
- the file name of epoch 1 and epoch 2 in each scene folder (e.g. `pc0.ply`, `pc1.ply`)
- the name of the change-label field on epoch 2 (e.g. `label_ch`)
- which code means "unchanged" and which means "mobile objects"

To print the fields of a PLY file:
```
python3 -c "import life4d_pilot as p; d=p.read_ply('SCENE/pc1.ply'); print(list(d), {k: sorted(set(d[k].astype(int)))[:10] for k in d if 'label' in k})"
```

## 4. Run
```
python3 life4d_pilot.py --root /path/to/Urb3DCD-v2 --a-pattern "pc0*" --b-pattern "pc1*" \
   --label-field label_ch --unchanged 0 --transient <mobile-code> --dataset-name "Urb3DCD-v2" --out results
```
First test the kit itself: `python3 make_test_pairs.py /tmp/test 12` and run the same command on `/tmp/test`.
Numbers from the test data are NOT results; never report them.

## 5. Use the outputs
- Copy `results/pilot_table_c1.tex` and `results/pilot_sentence.tex` next to `FT270100174_C1_v5.tex` and recompile:
  Table 3 gains the real-LiDAR rows and Section 7 gains the sentence. It still fits 7 pages.
- Copy `results/pilot_table.tex` and `results/pilot_sentence.tex` next to `preprint/life4d_preprint.tex` and recompile.

## 6. Read the numbers honestly before you use them
- Changes are rare in real data, so "keep last observation" may score close to Life4D on overall error.
  The expected advantages are calibration (ECE) and **re-survey efficiency on unobserved cells**
  (share of real changes found in the top-20% risk cells; random = 20%).
- If Life4D does not beat the baselines, say so in the preprint and keep the simulation as the evidence. A weak honest
  result is safer than a strong claim an assessor can test.
- Check the class codes: a wrong "unchanged" code silently breaks the ground truth.

## 7. Post the preprint
Fill the red boxes (repository URL, acknowledgements including any AI-assisted drafting/coding), check every number,
post to arXiv (cs.CV), then add it to B11 and to "Why I am best placed" in C1.
