#!/usr/bin/env python3
"""Make small synthetic bi-temporal ALS scenes (binary PLY, Urb3DCD-like fields) to test life4d_pilot.py.
Labels on epoch B: 0 unchanged, 1 new building, 2 demolition, 3 new vegetation, 4 vegetation growth,
5 missing vegetation, 7 mobile object. Usage: python3 make_test_pairs.py OUTDIR [n_scenes]"""
import os, sys
import numpy as np

def write_ply(path, cols):
    names = list(cols); n = len(cols[names[0]])
    dt = np.dtype([(k, "<f4") for k in names])
    arr = np.empty(n, dt)
    for k in names: arr[k] = cols[k]
    with open(path, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % n).encode())
        for k in names: f.write(f"property float {k}\n".encode())
        f.write(b"end_header\n"); f.write(arr.tobytes())

def scene(rng, S=120.0, dens=4.0):
    objs = []
    for _ in range(rng.integers(6, 12)):
        x, y = rng.uniform(5, S - 25, 2); w, d = rng.uniform(8, 20, 2); h = rng.uniform(5, 25)
        objs.append(("bld", x, y, w, d, h))
    for _ in range(rng.integers(10, 25)):
        objs.append(("veg", *rng.uniform(5, S - 5, 2), rng.uniform(2, 5), 0, rng.uniform(5, 14)))
    return objs

def sample(objs, rng, S, dens, changes=None):
    n = int(S * S * dens); x = rng.uniform(0, S, n); y = rng.uniform(0, S, n); z = rng.normal(0, 0.05, n); lab = np.zeros(n)
    for k, o in enumerate(objs):
        kind, ox, oy, a, b, h = o
        ch = changes.get(k) if changes else None
        if kind == "bld":
            m = (x > ox) & (x < ox + a) & (y > oy) & (y < oy + b)
            if ch == "demolish": lab[m] = 2; continue
            z[m] = h + rng.normal(0, 0.05, m.sum())
        else:
            r = a * (1.4 if ch == "grow" else 1.0); hh = h * (1.3 if ch == "grow" else 1.0)
            m = (x - ox) ** 2 + (y - oy) ** 2 < r * r
            if ch == "remove": lab[m] = 5; continue
            z[m] = hh * rng.uniform(0.6, 1.0, m.sum())
            if ch == "grow": lab[m] = 4
            if ch == "new": lab[m] = 3
    if changes and "new_bld" in changes:
        ox, oy, a, b, h = changes["new_bld"]
        m = (x > ox) & (x < ox + a) & (y > oy) & (y < oy + b); z[m] = h; lab[m] = 1
    car = (np.abs(x - rng.uniform(0, S)) < 2.5) & (np.abs(y - rng.uniform(0, S)) < 1.2)
    if changes is not None: z[car] = 1.5; lab[car] = 7
    return x, y, z, lab

def main():
    out = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    rng = np.random.default_rng(3); S = 120.0
    for s in range(n):
        objs = scene(rng, S)
        ch = {}
        for k, o in enumerate(objs):
            r = rng.random()
            if o[0] == "bld" and r < 0.15: ch[k] = "demolish"
            if o[0] == "veg" and r < 0.25: ch[k] = "grow"
            elif o[0] == "veg" and r < 0.35: ch[k] = "remove"
        if rng.random() < 0.7: ch["new_bld"] = (*rng.uniform(10, S - 30, 2), 15, 12, rng.uniform(8, 30))
        d = os.path.join(out, f"scene_{s:02d}"); os.makedirs(d, exist_ok=True)
        xa, ya, za, la = sample(objs, rng, S, 4.0, None)
        xb, yb, zb, lb = sample(objs, rng, S, 4.0, ch)
        write_ply(os.path.join(d, "pc0.ply"), dict(x=xa, y=ya, z=za, label_ch=la))
        write_ply(os.path.join(d, "pc1.ply"), dict(x=xb, y=yb, z=zb, label_ch=lb))
    print("wrote", n, "scenes to", out)

if __name__ == "__main__":
    main()
