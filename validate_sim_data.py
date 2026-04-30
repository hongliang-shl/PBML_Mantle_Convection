"""
Validate that a simulation directory follows the format expected by datasetio.py.

Usage:
    python validate_sim_data.py path/to/data_dir [sim_id]

Examples:
    python validate_sim_data.py my_TPH                 # checks sims.pt and ALL sims
    python validate_sim_data.py my_TPH 0               # checks only sim_0 (in any split)
    python validate_sim_data.py my_TPH train/sim_0     # checks specific path

The script returns exit code 0 on success, non-zero on hard errors.
Soft warnings are printed but do not fail the script.
"""

import os
import sys
import numpy as np
import torch


GR = 128
NX = 506
EPS = 1e-3


class V:
    """Validator with hard errors and soft warnings."""

    def __init__(self):
        self.errors = []
        self.warnings = []

    def err(self, msg):
        self.errors.append(msg)
        print(f"  [ERR ] {msg}")

    def warn(self, msg):
        self.warnings.append(msg)
        print(f"  [WARN] {msg}")

    def ok(self, msg):
        print(f"  [ OK ] {msg}")


def check_tensor(v: V, t, name, expected_shape=None, dtype=torch.float64):
    if not torch.is_tensor(t):
        v.err(f"{name}: not a torch.Tensor (got {type(t).__name__})")
        return False
    if dtype is not None and t.dtype != dtype:
        v.warn(f"{name}: dtype is {t.dtype}, expected {dtype} (will be cast at load)")
    if expected_shape is not None and tuple(t.shape) != tuple(expected_shape):
        v.err(f"{name}: shape={tuple(t.shape)}, expected {tuple(expected_shape)}")
        return False
    if torch.isnan(t).any():
        v.err(f"{name}: contains NaN")
        return False
    if torch.isinf(t).any():
        v.err(f"{name}: contains Inf")
        return False
    return True


def check_sim(sim_dir: str, v: V):
    print(f"\n=== {sim_dir} ===")

    if not os.path.isdir(sim_dir):
        v.err(f"directory does not exist")
        return

    # Required files
    required = [
        "xc.pt", "yc.pt", "times.pt",
        "e1_Tprev_data_select_snaps.pt",
        "e1_uprev_data_select_snaps.pt",
        "e1_vprev_data_select_snaps.pt",
    ]
    for f in required:
        p = os.path.join(sim_dir, f)
        if os.path.isfile(p):
            sz = os.path.getsize(p) / 1024 / 1024
            v.ok(f"{f} present ({sz:.2f} MB)")
        else:
            v.err(f"missing required file: {f}")
            return

    # Coordinates
    xc = torch.load(os.path.join(sim_dir, "xc.pt"), weights_only=False)
    yc = torch.load(os.path.join(sim_dir, "yc.pt"), weights_only=False)
    if not check_tensor(v, xc, "xc.pt", (GR, NX)):
        return
    if not check_tensor(v, yc, "yc.pt", (GR, NX)):
        return
    # Coordinate orientation
    if abs(yc[0, 0].item() - 0.0) > 0.05:
        v.err(f"yc[0, 0] = {yc[0, 0]:.4f}, expected ~0 (bottom). "
              f"Did you flip the y axis?")
    if abs(yc[-1, 0].item() - 1.0) > 0.05:
        v.err(f"yc[-1, 0] = {yc[-1, 0]:.4f}, expected ~1 (top)")
    if abs(xc[0, 0].item() - 0.0) > 0.05:
        v.err(f"xc[0, 0] = {xc[0, 0]:.4f}, expected ~0 (left)")
    if abs(xc[0, -1].item() - 4.0) > 0.05:
        v.err(f"xc[0, -1] = {xc[0, -1]:.4f}, expected ~4 (right, AR=4)")
    v.ok(f"xc range [{xc.min():.4f}, {xc.max():.4f}], "
         f"yc range [{yc.min():.4f}, {yc.max():.4f}]")

    # Times
    times = torch.load(os.path.join(sim_dir, "times.pt"), weights_only=False)
    if times.dim() != 1:
        v.err(f"times.pt should be 1-D, got shape {tuple(times.shape)}")
        return
    if (torch.diff(times) <= 0).any():
        v.err("times.pt is not strictly increasing")
    if times.min() < 0:
        v.warn(f"times.pt has negative entries (min={times.min():.3e})")
    v.ok(f"times: {len(times)} steps, t in [{times.min():.3e}, {times.max():.3e}]")

    # Fields
    T = torch.load(os.path.join(sim_dir, "e1_Tprev_data_select_snaps.pt"), weights_only=False)
    u = torch.load(os.path.join(sim_dir, "e1_uprev_data_select_snaps.pt"), weights_only=False)
    vfld = torch.load(os.path.join(sim_dir, "e1_vprev_data_select_snaps.pt"), weights_only=False)
    if not check_tensor(v, T, "Tprev", None):
        return
    n_snap = T.shape[0]
    if not check_tensor(v, T, "Tprev", (n_snap, 1, GR, NX)):
        return
    if not check_tensor(v, u, "uprev", (n_snap, 1, GR, NX)):
        return
    if not check_tensor(v, vfld, "vprev", (n_snap, 1, GR, NX)):
        return
    v.ok(f"snap fields shape ({n_snap}, 1, {GR}, {NX})")

    if n_snap > len(times):
        v.err(f"snaps has {n_snap} frames but times.pt has only {len(times)} steps")

    # T physical sanity
    if T.min() < -0.1 or T.max() > 1.5:
        v.err(f"T range [{T.min():.4f}, {T.max():.4f}] "
              f"is far outside Boussinesq [0, 1]; check non-dimensionalization")
    elif T.min() < 0 or T.max() > 1.05:
        v.warn(f"T range [{T.min():.4f}, {T.max():.4f}] slightly out of [0, 1.05]")
    else:
        v.ok(f"T range [{T.min():.4f}, {T.max():.4f}] OK")

    # T boundary
    T_bot_mean = T[:, 0, 0, :].mean().item()
    T_top_mean = T[:, 0, -1, :].mean().item()
    if abs(T_bot_mean - 1.0) > 0.05:
        v.err(f"T[bottom] mean = {T_bot_mean:.4f}, expected ~1 "
              f"(remember: y_idx=0 is bottom in this codebase)")
    if abs(T_top_mean) > 0.05:
        v.err(f"T[top] mean = {T_top_mean:.4f}, expected ~0")

    # Velocity sanity
    uvmax = max(u.abs().max().item(), vfld.abs().max().item())
    if uvmax < 1e-3:
        v.err(f"velocity max abs = {uvmax:.3e}, suspicious "
              f"(forgot to save? convection not started?)")
    elif uvmax > 1e7:
        v.err(f"velocity max abs = {uvmax:.3e}, suspicious "
              f"(did you save dimensional velocity in m/s instead of "
              f"non-dimensional?)")
    else:
        v.ok(f"velocity max abs = {uvmax:.3e}")

    # Free-slip top (most strict because face boundary)
    v_top = vfld[:, 0, -1, :].abs().mean().item()
    if v_top > 0.1 * uvmax:
        v.warn(f"v[top] mean abs = {v_top:.3e} not small relative to "
               f"|v|max={uvmax:.3e}; check free-slip BC")

    # Divergence-freeness (single frame, central differences)
    f0u, f0v = u[0, 0], vfld[0, 0]
    du_dx = (f0u[:, 2:] - f0u[:, :-2]) / 2.0
    dv_dy = (f0v[2:, :] - f0v[:-2, :]) / 2.0
    div = du_dx[1:-1, :] + dv_dy[:, 1:-1]
    rel_div = (div.abs().mean() / max(uvmax, 1e-12)).item()
    if rel_div > 1e-2:
        v.err(f"|∇·u|_mean / |u|_max = {rel_div:.3e}, "
              f"expected <1e-2 (incompressible Stokes); "
              f"check that you saved cell-centered velocities")
    else:
        v.ok(f"|∇·u|_mean / |u|_max = {rel_div:.3e}")


def check_root(data_dir: str, v: V, target_sim=None):
    sims_path = os.path.join(data_dir, "sims.pt")
    if not os.path.isfile(sims_path):
        v.err(f"missing {sims_path}")
        return

    sims = torch.load(sims_path, weights_only=False)
    if not isinstance(sims, list):
        v.err(f"sims.pt should be a list, got {type(sims)}")
        return

    print(f"\n=== sims.pt root metadata ===")
    print(f"  total entries: {len(sims)}")
    by_split = {}
    for s in sims:
        by_split.setdefault(s[1], []).append(s)
    for split, items in by_split.items():
        print(f"  {split}: {len(items)} sims")

    # Schema check
    for s in sims[:3]:
        if len(s) < 7:
            v.err(f"sims entry too short: {s}")
            return
        num, ds, raq, fkt, fkp, gr, ar = s[:7]
        if not isinstance(num, int):
            v.warn(f"sim num should be int, got {type(num).__name__}")
        if ds not in ("train", "cv", "test"):
            v.err(f"unknown split '{ds}', expected train/cv/test")
        if gr != GR:
            v.warn(f"sim {num}: gr={gr}, code expects {GR}")
        if ar != 4:
            v.warn(f"sim {num}: ar={ar}, code expects 4")

    # Check individual sims
    if target_sim is not None:
        check_sim(os.path.join(data_dir, target_sim) if "/" in target_sim
                  else None, v)
        for s in sims:
            if str(s[0]) == str(target_sim):
                check_sim(os.path.join(data_dir, s[1], f"sim_{s[0]}"), v)
                return
    else:
        for s in sims:
            check_sim(os.path.join(data_dir, s[1], f"sim_{s[0]}"), v)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)

    data_dir = sys.argv[1]
    target = sys.argv[2] if len(sys.argv) > 2 else None
    v = V()

    if target and "/" in target:
        # Direct path
        check_sim(os.path.join(data_dir, target), v)
    else:
        check_root(data_dir, v, target)

    print("\n" + "=" * 60)
    print(f"Errors  : {len(v.errors)}")
    print(f"Warnings: {len(v.warnings)}")
    print("=" * 60)
    sys.exit(1 if v.errors else 0)
