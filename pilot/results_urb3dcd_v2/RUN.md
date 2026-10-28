# Urb3DCD-v2 pilot run (28 Oct 2026)
Data: Urb3DCD-v2, IEEE_Dataset_V2_Lid05_MS, subset 1-Lidar05, Test scenes LyonS, LyonS3, LyonS4.
Label codes (label_ch): 0 unchanged, 6 mobile objects (transient); label_mono on epoch A: 0 ground, 1 building, 2 vegetation, 3 mobile.
Command:
    python3 life4d_pilot.py --root urb3dcd_pilot --a-pattern "pointCloud0*" --b-pattern "pointCloud1*" \
      --label-field label_ch --unchanged 0 --transient 6 --cell 3 --seeds 10 --class-field label_mono \
      --dataset-name "Urb3DCD-v2" --out results
Sensitivity: --cell 2 / 4, and without --class-field (label-free class): ordering of combined error and ECE unchanged.
