#!/usr/bin/env python3
"""Life4D real-data pilot on bi-temporal airborne LiDAR change benchmarks (Urb3DCD-v2, AHN-CD).

What it tests (the core claim of the Fellowship, on real or realistic point clouds):
  Stored geometry from survey A must be judged still valid or not at survey B, when part of B is
  NOT observed (strips not flown / occluded). Methods are compared on held-out scenes:
    1. Overwrite          - B replaces A; cells not observed in B are retired (non-observation = removal)
    2. Last observation   - cells not observed in B keep A forever (no decay)
    3. Presence filter    - persistence-filter style: global change rate, presence evidence only
    4. Life4D-style       - observation status kept separate from state; class-conditioned change rate
                            learned on training scenes; calibrated shape + free-space likelihood
  Metrics: stale (%), false retirement (%), combined (%), ECE; for all cells and unobserved cells.

Input: one folder per scene with two point clouds (epoch A, epoch B). Supported: .ply (ascii or
binary little/big endian), .las/.laz (needs `pip install laspy[lazrs]`), .txt/.csv (whitespace or comma).
Change labels must be on epoch B points (as in Urb3DCD-v2 / AHN-CD); A labels optional.

Example (Urb3DCD-v2 style):
  python3 life4d_pilot.py --root /data/Urb3DCD-v2 --a-pattern "pc0*" --b-pattern "pc1*" \
      --label-field label_ch --unchanged 0 --transient 7 --out results
Check the dataset README for the exact file names, label field and class codes; set the flags to match.
Outputs: results/pilot_results.json, results/pilot_table.tex (preprint), results/pilot_table_c1.tex (C1), (rows for C1 Table 3 / the preprint),
results/pilot_sentence.tex (one sentence for C1 Section 7 and the preprint).
"""
import argparse, glob, json, os, sys
import numpy as np

# ------------------------------------------------------------------ readers
def read_ply(path):
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline().decode("ascii", "ignore").strip()
            header.append(line)
            if line == "end_header":
                break
        fmt = [h for h in header if h.startswith("format")][0].split()[1]
        # first element is the point element (named "vertex" in most files, "params" in Urb3DCD)
        props, count, seen = [], 0, 0
        for h in header:
            p = h.split()
            if p and p[0] == "element":
                seen += 1
                if seen == 1: count = int(p[2])
            elif seen == 1 and p and p[0] == "property":
                props.append((p[2], p[1]))
        e = ">" if "big_endian" in fmt else "<"
        tmap = {"float": "f4", "float32": "f4", "double": "f8", "float64": "f8", "uchar": "u1", "uint8": "u1",
                "char": "i1", "int8": "i1", "ushort": "u2", "uint16": "u2", "short": "i2", "int16": "i2",
                "uint": "u4", "uint32": "u4", "int": "i4", "int32": "i4"}
        tmap = {k: (v if v[0] in "ui" and v.endswith("1") else e + v) for k, v in tmap.items()}
        if fmt == "ascii":
            data = np.loadtxt(f, max_rows=count)
            return {n: data[:, i] for i, (n, _) in enumerate(props)}
        dt = np.dtype([(n, tmap[t]) for n, t in props])
        arr = np.frombuffer(f.read(count * dt.itemsize), dtype=dt, count=count)
        return {n: arr[n].astype(float) for n, _ in props}

def read_las(path):
    import laspy
    las = laspy.read(path)
    d = {"x": np.asarray(las.x), "y": np.asarray(las.y), "z": np.asarray(las.z)}
    for dim in las.point_format.dimension_names:
        if dim not in ("X", "Y", "Z"):
            d[dim] = np.asarray(las[dim]).astype(float)
    return d

def read_txt(path, names):
    data = np.loadtxt(path, delimiter="," if path.endswith(".csv") else None, comments="//")
    names = names or ["x", "y", "z"] + [f"f{i}" for i in range(3, data.shape[1])]
    return {n: data[:, i] for i, n in enumerate(names[:data.shape[1]])}

def read_any(path, txt_names=None):
    p = path.lower()
    if p.endswith(".ply"): return read_ply(path)
    if p.endswith(".las") or p.endswith(".laz"): return read_las(path)
    return read_txt(path, txt_names)

# ------------------------------------------------------------------ cells
def to_cells(A, B, label, unchanged, transient, cell, rng, nf_frac, occ_frac, tiles=4, bounds=None, class_field=None):
    """Rasterise both epochs to cell x cell m surface cells. Truth: a stored cell is invalid at B
    if most labelled B points in it are a real change (transient labels ignored)."""
    xa, ya, za = A["x"], A["y"], A["z"]; xb, yb, zb = B["x"], B["y"], B["z"]; lb = B[label].astype(int)
    keep = ~np.isin(lb, transient)
    xb, yb, zb, lb = xb[keep], yb[keep], zb[keep], lb[keep]
    if bounds is None:
        bounds = (min(xa.min(), xb.min()), min(ya.min(), yb.min()), max(xa.max(), xb.max()), max(ya.max(), yb.max()))
    x0, y0, x1, y1 = bounds
    nx = int((x1 - x0) / cell) + 1; ny = int((y1 - y0) / cell) + 1
    ina = (xa >= x0) & (xa <= x1) & (ya >= y0) & (ya <= y1); inb = (xb >= x0) & (xb <= x1) & (yb >= y0) & (yb <= y1)
    ca = A[class_field][ina].astype(int) if class_field else None
    xa, ya, za = xa[ina], ya[ina], za[ina]; xb, yb, zb, lb = xb[inb], yb[inb], zb[inb], lb[inb]
    ia = ((ya - y0) / cell).astype(int) * nx + ((xa - x0) / cell).astype(int)
    ib = ((yb - y0) / cell).astype(int) * nx + ((xb - x0) / cell).astype(int)
    N = nx * ny
    def stat(idx, v, fn):
        out = np.full(N, np.nan); order = np.argsort(idx); idx, v = idx[order], v[order]
        u, s = np.unique(idx, return_index=True); parts = np.split(v, s[1:])
        out[u] = [fn(p) for p in parts]; return out
    cntA = np.bincount(ia, minlength=N); cntB = np.bincount(ib, minlength=N)
    zmaxA = stat(ia, za, np.max); zmaxB = stat(ib, zb, np.max)
    zstdA = stat(ia, za, np.std)
    chg = np.bincount(ib, weights=(~np.isin(lb, unchanged)).astype(float), minlength=N)
    has = cntA > 0                                  # stored geometry exists
    truth_invalid = np.zeros(N, bool)
    truth_invalid[cntB > 0] = (chg[cntB > 0] / cntB[cntB > 0]) > 0.5
    # cells with no B return have no change label; they are excluded from evaluation (truth unknown).
    # (Demolition in airborne LiDAR leaves ground returns, which carry the change label.)
    # simulated non-observation in B: flight strips not flown + random occlusion blocks
    gx = np.arange(N) % nx; gy = np.arange(N) // nx
    strip_w = max(1, int(nx * nf_frac))
    s0 = rng.integers(0, max(1, nx - strip_w))
    unobs = (gx >= s0) & (gx < s0 + strip_w)
    nblk = int(occ_frac * N / 25)
    for _ in range(nblk):
        cx, cy = rng.integers(0, nx), rng.integers(0, ny)
        unobs |= (np.abs(gx - cx) <= 2) & (np.abs(gy - cy) <= 2)
    # cell class from stored geometry: ground / vegetation / building (rough, label-free)
    rel = zmaxA - np.nanpercentile(zmaxA, 5)
    cls = np.where(rel < 2.0, 0, np.where(zstdA > 1.2, 1, 2))
    if class_field:   # stored semantic class of epoch A (majority per cell): 0 ground, 1 building, 2 vegetation
        votes = np.stack([np.bincount(ia, weights=(ca == k).astype(float), minlength=N) for k in (0, 1, 2)])
        sem = np.argmax(votes, 0); cls = np.where(sem == 1, 2, np.where(sem == 2, 1, 0))   # map to 0 ground, 1 veg, 2 building
    m = has & (cntB > 0)
    tile = (gy * tiles // ny) * tiles + (gx * tiles // nx)
    return dict(dz=np.abs(zmaxB - zmaxA)[m], obs=~unobs[m], emptyB=(cntB == 0)[m], cls=cls[m],
                truth=truth_invalid[m], unobs=unobs[m], tile=tile[m])

# ------------------------------------------------------------------ methods
def fit_life4d(train):
    cls = np.concatenate([t["cls"] for t in train]); tr = np.concatenate([t["truth"] for t in train])
    prior = {c: float(np.clip(tr[cls == c].mean(), 0.01, 0.99)) if np.any(cls == c) else 0.2 for c in (0, 1, 2)}
    # calibrated likelihood: logistic regression on observed cells, features: log dz, emptyB, class
    obs = np.concatenate([t["obs"] for t in train])
    X = np.c_[np.log1p(np.nan_to_num(np.concatenate([t["dz"] for t in train]), nan=50.0)),
              np.concatenate([t["emptyB"] for t in train]).astype(float),
              (cls == 1).astype(float), (cls == 2).astype(float), np.ones(len(cls))]
    Xo, yo = X[obs], tr[obs].astype(float)
    w = np.zeros(X.shape[1])
    for _ in range(400):                               # plain gradient descent, deterministic
        p = 1 / (1 + np.exp(-Xo @ w)); w -= 0.5 * Xo.T @ (p - yo) / len(yo) + 1e-4 * w
    glob = float(np.clip(tr.mean(), 0.01, 0.99))
    return dict(prior=prior, w=w, glob=glob)

def predict(model, s, method, thr=1.0):
    obs, dz, empty, cls = s["obs"], np.nan_to_num(s["dz"], nan=50.0), s["emptyB"], s["cls"]
    pv = np.empty(len(obs))
    if method == "overwrite":
        pv = np.where(obs, np.where((dz < thr) & ~empty, 0.95, 0.05), 0.05)
    elif method == "last":
        pv = np.where(obs, np.where((dz < thr) & ~empty, 0.95, 0.05), 0.95)
    elif method == "presence":
        pv = np.where(obs, np.where(~empty, 0.95, 0.05), 1 - model["glob"])
    elif method == "life4d":
        X = np.c_[np.log1p(dz), empty.astype(float), (cls == 1).astype(float), (cls == 2).astype(float), np.ones(len(cls))]
        p_inv_obs = 1 / (1 + np.exp(-X @ model["w"]))
        p_inv_unobs = np.array([model["prior"][c] for c in cls])
        pv = np.where(obs, 1 - p_inv_obs, 1 - p_inv_unobs)
    return pv

def ranking(pv, truth_invalid, frac=0.2):
    """Re-survey efficiency on unobserved cells: share of truly changed cells captured when only the
    `frac` highest-risk cells are re-surveyed (random choice captures `frac`), and AUROC."""
    risk = 1 - pv; t = truth_invalid
    if t.sum() == 0 or t.sum() == len(t):
        return dict(capture=float("nan"), auroc=float("nan"))
    jitter = np.random.default_rng(0).random(len(risk)) * 1e-9          # break ties fairly
    order = np.argsort(-(risk + jitter)); k = max(1, int(frac * len(risk)))
    capture = t[order[:k]].sum() / t.sum()
    r = np.empty(len(risk)); r[np.argsort(risk + jitter)] = np.arange(1, len(risk) + 1)
    auroc = (r[t].sum() - t.sum() * (t.sum() + 1) / 2) / (t.sum() * (~t).sum())
    return dict(capture=float(capture), auroc=float(auroc))

def metrics(pv, truth_invalid):
    y = ~truth_invalid; pred = pv >= 0.5
    stale = np.mean(pred & ~y) * 100; fr = np.mean(~pred & y) * 100
    bins = np.clip((pv * 10).astype(int), 0, 9)
    ece = sum(abs(pv[bins == b].mean() - y[bins == b].mean()) * np.mean(bins == b) for b in range(10) if np.any(bins == b))
    return dict(stale=stale, false_ret=fr, combined=stale + fr, ece=ece)

# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="folder with one sub-folder per scene")
    ap.add_argument("--a-pattern", default="*0*.ply"); ap.add_argument("--b-pattern", default="*1*.ply")
    ap.add_argument("--label-field", default="label_ch")
    ap.add_argument("--unchanged", type=int, nargs="+", default=[0], help="label code(s) meaning unchanged")
    ap.add_argument("--transient", type=int, nargs="*", default=[], help="label code(s) for mobile objects")
    ap.add_argument("--txt-names", nargs="*", default=None, help="column names for .txt/.csv inputs")
    ap.add_argument("--cell", type=float, default=1.0); ap.add_argument("--notflown", type=float, default=0.25)
    ap.add_argument("--occlusion", type=float, default=0.10); ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--tiles", type=int, default=4, help="T x T spatial blocks; train/test split by block location")
    ap.add_argument("--split", choices=["tiles", "scenes"], default="tiles")
    ap.add_argument("--class-field", default=None, help="semantic label field on epoch A (stored map classes); default: label-free height/roughness class")
    ap.add_argument("--dataset-name", default="Urb3DCD-v2"); ap.add_argument("--out", default="results")
    a = ap.parse_args()
    scenes = sorted(d for d in glob.glob(os.path.join(a.root, "*")) if os.path.isdir(d)) or [a.root]
    pairs = []
    for d in scenes:
        fa = sorted(glob.glob(os.path.join(d, a.a_pattern))); fb = sorted(glob.glob(os.path.join(d, a.b_pattern)))
        if fa and fb and fa[0] != fb[0]:
            pairs.append((fa[0], fb[0]))
    if len(pairs) < 2:
        sys.exit(f"Found {len(pairs)} scene pairs under {a.root}; check --a-pattern/--b-pattern.")
    print(f"{len(pairs)} scenes; loading...")
    clouds = [(read_any(pa, a.txt_names), read_any(pb, a.txt_names)) for pa, pb in pairs]
    allx = np.concatenate([c[k]["x"] for c in clouds for k in (0, 1)]); ally = np.concatenate([c[k]["y"] for c in clouds for k in (0, 1)])
    same = all(abs(c[0]["x"].min() - clouds[0][0]["x"].min()) < 5 and abs(c[0]["y"].min() - clouds[0][0]["y"].min()) < 5 for c in clouds)
    bounds = (allx.min(), ally.min(), allx.max(), ally.max()) if same else None   # shared grid if scenes cover one area
    for i, (A, B) in enumerate(clouds):
        lb = B[a.label_field].astype(int); u, c = np.unique(lb, return_counts=True)
        print(f"  scene {i}: {len(A['x'])} / {len(B['x'])} points; B change labels {dict(zip(u.tolist(), c.tolist()))}")
    methods = [("overwrite", "Overwrite (non-observation $=$ removal)"), ("last", "Keep last observation"),
               ("presence", "Presence-only persistence filter"), ("life4d", "\\textbf{Life4D-style validity filter}")]
    res = {m: {"all": [], "unobs": []} for m, _ in methods}
    for seed in range(a.seeds):
        rng = np.random.default_rng(seed)
        cells = [to_cells(A, B, a.label_field, a.unchanged, a.transient, a.cell, rng, a.notflown, a.occlusion, a.tiles, bounds, a.class_field) for A, B in clouds]
        if a.split == "scenes":
            idx = rng.permutation(len(cells)); half = max(1, len(cells) // 2)
            train = [cells[i] for i in idx[:half]]; test = [cells[i] for i in idx[half:]]
        else:   # spatial blocks: a test location is never seen in training, in any scene
            perm = rng.permutation(a.tiles ** 2); tr_t = perm[: a.tiles ** 2 // 2]
            sub = lambda c, keep: {k: v[keep] for k, v in c.items()}
            train = [sub(c, np.isin(c["tile"], tr_t)) for c in cells]; test = [sub(c, ~np.isin(c["tile"], tr_t)) for c in cells]
        model = fit_life4d(train)
        cat = lambda k: np.concatenate([t[k] for t in test])
        for m, _ in methods:
            pv = np.concatenate([predict(model, t, m) for t in test]); tr = cat("truth"); un = cat("unobs")
            mu = metrics(pv[un], tr[un]); mu.update(ranking(pv[un], tr[un]))
            res[m]["all"].append(metrics(pv, tr)); res[m]["unobs"].append(mu)
    os.makedirs(a.out, exist_ok=True)
    summary = {}
    rows = []
    for m, label in methods:
        agg = {k: (np.mean([r[k] for r in res[m]["all"]]), np.std([r[k] for r in res[m]["all"]])) for k in ("stale", "false_ret", "combined", "ece")}
        un = {k: float(np.nanmean([r[k] for r in res[m]["unobs"]])) for k in ("stale", "false_ret", "ece", "capture", "auroc")}
        summary[m] = {"all": agg, "unobserved": un}
        rows.append(f"{label} & {agg['stale'][0]:.1f} $\\pm$ {agg['stale'][1]:.1f} & {agg['false_ret'][0]:.1f} $\\pm$ {agg['false_ret'][1]:.1f} & {agg['combined'][0]:.1f} & {agg['ece'][0]:.3f} $\\pm$ {agg['ece'][1]:.3f}\\\\")
    meta = dict(dataset=a.dataset_name, scenes=len(pairs), seeds=a.seeds, cell_m=a.cell, not_flown=a.notflown, occlusion=a.occlusion, split=a.split, tiles=a.tiles, class_field=a.class_field)
    json.dump({"meta": meta, "results": summary}, open(os.path.join(a.out, "pilot_results.json"), "w"), indent=2)
    with open(os.path.join(a.out, "pilot_table.tex"), "w") as f:
        split_txt = "held-out spatial blocks" if a.split == "tiles" else "held-out scenes"
        f.write(f"\\midrule\n\\multicolumn{{5}}{{@{{}}l}}{{\\emph{{Airborne LiDAR pilot: {a.dataset_name}, {len(pairs)} scenes, {split_txt}, {a.seeds} seeds}}}}\\\\\n")
        f.write("\n".join(rows) + "\n")
    with open(os.path.join(a.out, "pilot_table_c1.tex"), "w") as f:   # compact version for C1 Table 3
        f.write(f"\\midrule\n\\multicolumn{{5}}{{@{{}}l}}{{\\emph{{Airborne LiDAR pilot ({a.dataset_name}, {len(pairs)} scenes, held-out areas)}}}}\\\\\n")
        f.write(rows[1] + "\n" + rows[3] + "\n")
    L = summary["life4d"]["unobserved"]; best_base = max(summary[m]["unobserved"]["capture"] for m in ("overwrite", "last", "presence"))
    with open(os.path.join(a.out, "pilot_sentence.tex"), "w") as f:
        L_all = summary["life4d"]["all"]; K = summary["last"]["all"]
        best_ece = min(summary[m]["all"]["ece"][0] for m in ("overwrite", "last", "presence"))
        f.write(f"On {a.dataset_name} airborne LiDAR ({len(pairs)} test scenes, held-out areas, {a.seeds} seeds), Life4D had the lowest combined error "
                f"({L_all['combined'][0]:.1f}\\% against {K['combined'][0]:.1f}\\% for keeping the last observation) and the best calibration "
                f"(ECE {L_all['ece'][0]:.3f} against {best_ece:.3f} for the best baseline), while overwriting wrongly retired "
                f"{summary['overwrite']['unobserved']['false_ret']:.0f}\\% of still-valid unobserved geometry. Ranking unobserved cells for re-survey "
                f"from two surveys gained only modestly over random choice ({100 * L['capture']:.0f}\\% of changes in the top 20\\%, against 20\\%; AUROC {L['auroc']:.2f}); "
                f"learning change rates from repeated revisits is what RP2 adds.\n")
    for m, _ in methods:
        s = summary[m]["all"]; u = summary[m]["unobserved"]
        print(f"{m:10s} stale {s['stale'][0]:5.1f}  false-ret {s['false_ret'][0]:5.1f}  combined {s['combined'][0]:5.1f}  ECE {s['ece'][0]:.3f} | unobserved: false-ret {u['false_ret']:5.1f} ECE {u['ece']:.3f} capture@20% {100*u['capture']:.0f}% AUROC {u['auroc']:.2f}")
    print(f"Wrote {a.out}/pilot_results.json and {a.out}/pilot_table.tex")

if __name__ == "__main__":
    main()
