#!/usr/bin/env python3
"""Life4D preliminary simulation study (FT270100174, v5).

Tests the core logic of validity-aware geometric memory on synthetic data
with a KNOWN, deliberately misspecified generative model:

  * entities in scenes, four classes with different change behaviour;
  * change times are Weibull (the filters assume exponential: misspecified);
  * a change is one of: modified, grown, replaced (geometry still present but
    no longer valid) or removed (geometry absent);
  * irregular revisits 2-14 months apart over 48 months; each entity is in
    view at a revisit with probability 0.6 (otherwise unobserved);
  * when in view: noisy presence evidence and a noisy geometric residual.

Methods (all output p(stored geometry still valid at query time)):
  1. Overwrite          - latest revisit replaces memory; unobserved = removed.
  2. Last observation   - keep last observed state forever (no decay).
  3. Presence filter    - persistence-filter style: binary presence evidence
                          only, one global exponential rate.
  4. Life4D-style       - separates observation status from state; class-
                          conditioned rates learned from interval-censored
                          revisit labels; presence + residual likelihood.

Second test (comparative anchoring): at sparsely observed sites (2 epochs),
compare a per-site rate (learned from that site alone) with a rate pooled
across like structures at other sites.

Usage: python3 life4d_sim.py   (numpy only; ~1 min; prints LaTeX-ready rows)
"""
import numpy as np

CLASSES = ["building", "road", "vegetation", "object"]
# Weibull shape k and scale (months) per class; change-type probabilities
WEIB = {"building": (1.6, 160.0), "road": (1.4, 90.0),
        "vegetation": (1.8, 40.0), "object": (1.2, 25.0)}
CTYPE = {  # modified, grown, replaced, removed
    "building": [0.6, 0.0, 0.2, 0.2], "road": [0.7, 0.0, 0.2, 0.1],
    "vegetation": [0.2, 0.6, 0.0, 0.2], "object": [0.2, 0.0, 0.2, 0.6]}
HORIZON, P_VIS = 48.0, 0.6
N_SCENES, N_PER_SCENE = 200, 40          # 8,000 entities


def simulate(rng, n_scenes=N_SCENES, sparse=False):
    ents = []
    for s in range(n_scenes):
        # irregular revisit schedule per scene
        t, times = 0.0, [0.0]
        while True:
            t += rng.uniform(2, 14)
            if t > HORIZON:
                break
            times.append(t)
        if sparse:
            times = times[:2]
        times = np.array(times)
        for _ in range(N_PER_SCENE):
            c = CLASSES[rng.integers(4)]
            k, lam = WEIB[c]
            tc = lam * rng.weibull(k)
            ctype = rng.choice(4, p=CTYPE[c])
            obs = []
            for tt in times[1:]:
                if rng.random() > P_VIS:
                    obs.append(None)
                    continue
                changed = tt >= tc
                removed = changed and ctype == 3
                present = (rng.random() < 0.95) if not removed else (rng.random() < 0.10)
                if changed and not removed:
                    resid = abs(rng.normal(2.2, 1.0))
                else:
                    resid = abs(rng.normal(0.0, 1.0))
                obs.append((present, resid))
            tq = rng.uniform(max(12.0, times[-1] if sparse else 12.0), HORIZON + 12)
            ents.append(dict(scene=s, c=c, tc=tc, ctype=ctype, times=times,
                             obs=obs, tq=tq, valid=tq < tc))
    return ents


def interval_labels(e):
    """(L, R): last revisit seen unchanged, first revisit seen changed (inf if never)."""
    L, R = 0.0, np.inf
    for tt, o in zip(e["times"][1:], e["obs"]):
        if o is None:
            continue
        if tt < e["tc"]:
            L = tt
        else:
            R = tt
            break
    return L, R


def fit_rate(intervals):
    """MLE of an exponential rate from interval-censored data (grid search)."""
    grid = np.exp(np.linspace(np.log(1e-4), np.log(1.0), 400))
    best, bl = grid[0], -np.inf
    for lam in grid:
        ll = 0.0
        for L, R in intervals:
            sL = np.exp(-lam * L)
            ll += np.log(max(sL - (np.exp(-lam * R) if np.isfinite(R) else 0.0), 1e-300))
        if ll > bl:
            best, bl = lam, ll
    return best


def presence_intervals(e):
    """Presence-only labels: only removal counts as change."""
    L, R = 0.0, np.inf
    for tt, o in zip(e["times"][1:], e["obs"]):
        if o is None:
            continue
        gone = tt >= e["tc"] and e["ctype"] == 3
        if not gone:
            L = tt
        else:
            R = tt
            break
    return L, R


def gauss(x, m, s):
    return np.exp(-0.5 * ((x - m) / s) ** 2) / s


def fit_lik(train):
    rv, rc, pv, pc = [], [], [], []
    for e in train:
        for tt, o in zip(e["times"][1:], e["obs"]):
            if o is None:
                continue
            (rc if tt >= e["tc"] else rv).append(o[1])
            (pc if tt >= e["tc"] else pv).append(o[0])
    rv, rc = np.array(rv), np.array(rc)
    return dict(mv=rv.mean(), sv=rv.std(), mc=rc.mean(), sc=rc.std(),
                pv=np.mean(pv), pc=np.mean(pc))


def filt(e, lam, lik=None, presence_only=False):
    """Two-state filter (valid / changed, changed absorbing) -> p(valid at tq)."""
    p, t_prev = 1.0, 0.0
    for tt, o in zip(e["times"][1:], e["obs"]):
        if tt > e["tq"]:
            break
        p *= np.exp(-lam * (tt - t_prev))
        t_prev = tt
        if o is None:
            continue                      # non-observation: no evidence
        pres, resid = o
        if presence_only:
            lv = 0.95 if pres else 0.05
            lc = 0.10 if pres else 0.90
        else:
            lv = (lik["pv"] if pres else 1 - lik["pv"]) * gauss(resid, lik["mv"], lik["sv"])
            lc = (lik["pc"] if pres else 1 - lik["pc"]) * gauss(resid, lik["mc"], lik["sc"])
        p = p * lv / (p * lv + (1 - p) * lc + 1e-300)
    return p * np.exp(-lam * (e["tq"] - t_prev))


def last_obs(e, overwrite=False):
    last = None
    for tt, o in zip(e["times"][1:], e["obs"]):
        if tt > e["tq"]:
            break
        if overwrite:
            last = o if o is not None else (False, 0.0)
        elif o is not None:
            last = o
    if last is None:
        return 0.95
    pres, resid = last
    return 0.95 if (pres and resid < 1.5) else 0.05


def metrics(p, y):
    p, y = np.asarray(p), np.asarray(y, bool)
    pred = p >= 0.5
    stale = np.mean(pred & ~y) * 100
    falseret = np.mean(~pred & y) * 100
    bins = np.clip((p * 10).astype(int), 0, 9)
    ece = sum(abs(p[bins == b].mean() - y[bins == b].mean()) * np.mean(bins == b)
              for b in range(10) if np.any(bins == b))
    return stale, falseret, stale + falseret, ece


def run_seed(seed):
    rng = np.random.default_rng(seed)
    ents = simulate(rng)
    scenes = rng.permutation(N_SCENES)
    tr_s = set(scenes[: N_SCENES // 2])
    train = [e for e in ents if e["scene"] in tr_s]
    test = [e for e in ents if e["scene"] not in tr_s]
    lam_c = {c: fit_rate([interval_labels(e) for e in train if e["c"] == c]) for c in CLASSES}
    lam_g = fit_rate([presence_intervals(e) for e in train])
    lik = fit_lik(train)
    y = [e["valid"] for e in test]
    out = {
        "Overwrite": metrics([last_obs(e, True) for e in test], y),
        "Last observation": metrics([last_obs(e) for e in test], y),
        "Presence filter": metrics([filt(e, lam_g, presence_only=True) for e in test], y),
        "Life4D-style": metrics([filt(e, lam_c[e["c"]], lik) for e in test], y),
    }
    # comparative anchoring: sparse sites (2 epochs), per-site vs pooled class rate
    sparse = simulate(rng, n_scenes=60, sparse=True)
    ps, pp, ys = [], [], []
    for s in range(60):
        site = [e for e in sparse if e["scene"] == s]
        lam_site = fit_rate([interval_labels(e) for e in site])   # one rate, own data only
        for e in site:
            ps.append(filt(e, lam_site, lik))
            pp.append(filt(e, lam_c[e["c"]], lik))
            ys.append(e["valid"])
    out["sparse_site"] = metrics(ps, ys)
    out["sparse_pooled"] = metrics(pp, ys)
    return out


if __name__ == "__main__":
    res = [run_seed(s) for s in range(10)]
    for k in res[0]:
        a = np.array([r[k] for r in res])
        m, sd = a.mean(0), a.std(0)
        print(f"{k:18s} & {m[0]:.1f} $\\pm$ {sd[0]:.1f} & {m[1]:.1f} $\\pm$ {sd[1]:.1f} "
              f"& {m[2]:.1f} & {m[3]:.3f} $\\pm$ {sd[3]:.3f} \\\\")
