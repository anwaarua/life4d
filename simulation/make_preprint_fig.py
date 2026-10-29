"""Reliability diagram + error bars from life4d_sim.py (seed 0..9)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import life4d_sim as L

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})
P = {k: [] for k in ("Overwrite", "Last observation", "Presence filter", "Life4D-style")}
Y = []
for seed in range(10):
    rng = np.random.default_rng(seed)
    ents = L.simulate(rng)
    scenes = rng.permutation(L.N_SCENES); tr_s = set(scenes[: L.N_SCENES // 2])
    train = [e for e in ents if e["scene"] in tr_s]; test = [e for e in ents if e["scene"] not in tr_s]
    lam_c = {c: L.fit_rate([L.interval_labels(e) for e in train if e["c"] == c]) for c in L.CLASSES}
    lam_g = L.fit_rate([L.presence_intervals(e) for e in train]); lik = L.fit_lik(train)
    Y += [e["valid"] for e in test]
    P["Overwrite"] += [L.last_obs(e, True) for e in test]
    P["Last observation"] += [L.last_obs(e) for e in test]
    P["Presence filter"] += [L.filt(e, lam_g, presence_only=True) for e in test]
    P["Life4D-style"] += [L.filt(e, lam_c[e["c"]], lik) for e in test]
Y = np.array(Y, bool)
fig, ax = plt.subplots(1, 1, figsize=(3.3, 2.6), dpi=300)
cols = {"Overwrite": "0.65", "Last observation": "#999966", "Presence filter": "#c08a3e", "Life4D-style": "#2f5fb0"}
ax.plot([0, 1], [0, 1], color="0.7", lw=0.7, ls="--")
for k, p in P.items():
    p = np.array(p); bins = np.clip((p * 10).astype(int), 0, 9)
    xs = [p[bins == b].mean() for b in range(10) if np.sum(bins == b) > 50]
    ys = [Y[bins == b].mean() for b in range(10) if np.sum(bins == b) > 50]
    ax.plot(xs, ys, marker="o", ms=3, lw=1.3 if k == "Life4D-style" else 0.9, color=cols[k], label=k)
ax.set_xlabel("predicted p(stored geometry still valid)"); ax.set_ylabel("observed frequency valid")
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(frameon=False, fontsize=6, loc="upper left")
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig("reliability.png")
print("ok")
