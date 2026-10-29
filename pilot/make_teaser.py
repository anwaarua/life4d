"""Scenario figure from Urb3DCD-v2 (LyonS): stored map, partial second survey,
true change, and what each method believes about the unobserved area."""
import sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Patch
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import life4d_pilot as P

ROOT = sys.argv[1] if len(sys.argv) > 1 else "urb3dcd_pilot"
CELL = 3.0
scenes = ["LyonS", "LyonS3", "LyonS4"]
clouds = {s: (P.read_ply(f"{ROOT}/{s}/pointCloud0.ply"), P.read_ply(f"{ROOT}/{s}/pointCloud1.ply")) for s in scenes}
allx = np.concatenate([c[k]["x"] for c in clouds.values() for k in (0, 1)])
ally = np.concatenate([c[k]["y"] for c in clouds.values() for k in (0, 1)])
bounds = (allx.min(), ally.min(), allx.max(), ally.max())

# model learned on the two other test scenes (different change realisations)
rng = np.random.default_rng(3)
train = [P.to_cells(*clouds[s], "label_ch", [0], [6], CELL, rng, 0.25, 0.10, 4, bounds, "label_mono") for s in scenes[1:]]
model = P.fit_life4d(train)

# grid for LyonS
A, B = clouds["LyonS"]
x0, y0, x1, y1 = bounds
nx = int((x1 - x0) / CELL) + 1; ny = int((y1 - y0) / CELL) + 1; N = nx * ny
lb = B["label_ch"].astype(int); keep = lb != 6
xb, yb, zb, lb = B["x"][keep], B["y"][keep], B["z"][keep], lb[keep]
ia = ((A["y"] - y0) / CELL).astype(int) * nx + ((A["x"] - x0) / CELL).astype(int)
ib = ((yb - y0) / CELL).astype(int) * nx + ((xb - x0) / CELL).astype(int)
cntA = np.bincount(ia, minlength=N); cntB = np.bincount(ib, minlength=N)
zA = np.full(N, np.nan); np.fmax.at(zA, ia, A["z"])
zB = np.full(N, np.nan); np.fmax.at(zB, ib, zb)
chg = np.bincount(ib, weights=(lb != 0).astype(float), minlength=N)
# dominant change class per cell (1 new bldg, 2 demolition, 3 new veg, 4 veg growth, 5 missing veg)
votes = np.stack([np.bincount(ib, weights=(lb == k).astype(float), minlength=N) for k in range(6)])
ccls = np.argmax(votes, 0)
truth = np.zeros(N, bool); m = cntB > 0; truth[m] = chg[m] / cntB[m] > 0.5
sem = np.argmax(np.stack([np.bincount(ia, weights=(A["label_mono"].astype(int) == k).astype(float), minlength=N) for k in (0, 1, 2)]), 0)
cls = np.where(sem == 1, 2, np.where(sem == 2, 1, 0))
gx = np.arange(N) % nx; gy = np.arange(N) // nx
# not-flown strip and occlusion blocks (fixed for the figure)
r2 = np.random.default_rng(11)
s0 = int(nx * 0.52); w = int(nx * 0.25)
unobs = (gx >= s0) & (gx < s0 + w)
for _ in range(int(0.10 * N / 25)):
    cx, cy = r2.integers(0, nx), r2.integers(0, ny)
    unobs |= (np.abs(gx - cx) <= 2) & (np.abs(gy - cy) <= 2)
valid_cells = (cntA > 0) & (cntB > 0)
dz = np.abs(zB - zA)
s = dict(obs=~unobs, dz=dz, emptyB=(cntB == 0), cls=cls)
pv = {k: P.predict(model, s, k) for k in ("overwrite", "last", "life4d")}

# crop window
X0, X1, Y0, Y1 = int(nx * 0.30), int(nx * 0.95), int(ny * 0.25), int(ny * 0.85)
def img(v, fill=np.nan):
    a = np.where(valid_cells, v, fill).astype(float).reshape(ny, nx)
    return a[Y0:Y1, X0:X1]
U = unobs.reshape(ny, nx)[Y0:Y1, X0:X1]
T = (truth & valid_cells).reshape(ny, nx)[Y0:Y1, X0:X1]

def err_stats(p):
    sel = unobs & valid_cells; y = ~truth[sel]; pred = p[sel] >= 0.5
    return 100 * np.mean(~pred & y), 100 * np.mean(pred & ~y)

fig, ax = plt.subplots(1, 5, figsize=(14.2, 2.85), gridspec_kw=dict(wspace=0.04))
fig.subplots_adjust(left=0.005, right=0.995, top=0.90, bottom=0.17)
ext = [0, (X1 - X0) * CELL, 0, (Y1 - Y0) * CELL]
hcm = plt.get_cmap("cividis")
zlo, zhi = np.nanpercentile(zA, 2), np.nanpercentile(zA, 99.5)

def outline(a):
    a.contour(U.astype(float), levels=[0.5], colors="#d62728", linewidths=0.9, origin="lower", extent=ext)

a = ax[0]; a.imshow(img(zA), origin="lower", cmap=hcm, vmin=zlo, vmax=zhi, extent=ext, interpolation="nearest")
a.set_title("(a) Stored map: survey 1", fontsize=10)
a = ax[1]; im = img(zB).copy(); im[U] = np.nan
a.set_facecolor("#d9d9d9"); a.imshow(im, origin="lower", cmap=hcm, vmin=zlo, vmax=zhi, extent=ext, interpolation="nearest")
outline(a); a.set_title("(b) Survey 2: grey = not seen", fontsize=10)
a = ax[2]
cc = np.where(valid_cells & truth, ccls, 0).reshape(ny, nx)[Y0:Y1, X0:X1].astype(float)
base = np.where(np.isnan(img(zA)), np.nan, 0.0)
ccol = ListedColormap(["#f0f0f0", "#e6550d", "#7f2704", "#31a354", "#a1d99b", "#756bb1"])
show = np.where(np.isnan(base), np.nan, cc)
a.imshow(show, origin="lower", cmap=ccol, vmin=-0.5, vmax=5.5, extent=ext, interpolation="nearest")
outline(a); a.set_title("(c) What really changed", fontsize=10)

vcm = LinearSegmentedColormap.from_list("v", ["#b2182b", "#f7f7f7", "#2166ac"])
titles = {"overwrite": "(d) Overwrite", "last": "Keep last observation", "life4d": "(e) Life4D: p(still valid)"}
for a, k in zip(ax[3:], ("overwrite", "life4d")):
    v = img(pv[k]); a.imshow(v, origin="lower", cmap=vcm, vmin=0, vmax=1, extent=ext, interpolation="nearest")
    outline(a)
    fr, st = err_stats(pv[k]); sel = unobs & valid_cells
    a.set_title(titles[k], fontsize=10)
    msg = (f"unseen area: {fr:.0f}% of still-valid\ngeometry wrongly retired" if k == "overwrite" else
           f"unseen area: mean p(valid) {pv[k][sel].mean():.2f};\nreally still valid: {100 * (~truth[sel]).mean():.0f}%")
    a.text(0.02, 0.02, msg, transform=a.transAxes, fontsize=7.5,
           va="bottom", bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
for a in ax:
    a.set_xticks([]); a.set_yticks([])
ax[0].plot([10, 110], [12, 12], "w-", lw=2); ax[0].text(60, 20, "100 m", color="w", ha="center", fontsize=7.5)

leg = [Patch(fc="#e6550d", label="new building"), Patch(fc="#7f2704", label="demolished"),
       Patch(fc="#31a354", label="new vegetation"), Patch(fc="#a1d99b", label="vegetation growth"),
       Patch(fc="#756bb1", label="vegetation removed"), Patch(fc="none", ec="#d62728", label="not re-surveyed")]
fig.legend(handles=leg, loc="lower left", bbox_to_anchor=(0.0, 0.0), ncol=6, fontsize=8, frameon=False)
cax = fig.add_axes([0.83, 0.085, 0.13, 0.035])
cb = plt.colorbar(plt.cm.ScalarMappable(cmap=vcm, norm=plt.Normalize(0, 1)), cax=cax, orientation="horizontal")
cb.set_ticks([0, 0.5, 1]); cb.ax.tick_params(labelsize=7); cb.ax.set_title("p(stored geometry still valid)", fontsize=7.5, pad=2)
fig.savefig("teaser.png", dpi=220, bbox_inches="tight")
fr_l, st_l = err_stats(pv["last"])
print("keep-last unseen:", round(fr_l, 1), round(st_l, 1))
for k in pv: print(k, [round(x, 1) for x in err_stats(pv[k])])
print("life4d mean p unseen", round(pv["life4d"][unobs & valid_cells].mean(),3), "last", round(pv["last"][unobs & valid_cells].mean(),3))
print("unseen changed %", round(100 * truth[unobs & valid_cells].mean(), 1))
