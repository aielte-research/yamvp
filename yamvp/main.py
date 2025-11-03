#!/usr/bin/env python3 
"""
Unified Venn plotter: supports N in {1,2,3,4,5} with a single entrypoint `venn(...)`.
- Geometry is created via helpers: _geom1, _geom2, _geom3, _geom4, _geom5.
- After geometry is loaded, rendering is generic and dimension-agnostic.
- For N=4, all original configurable geometry knobs are kept *inside* _geom4.
- For N=5, the geometry is five ellipses rotated at multiples of 72°, with per-ellipse
  translation along its own rotation and perpendicular to it, plus an ellipse ratio knob.
  Class labels are placed using a support-point method on each ellipse boundary:
  they sit just outside the outermost boundary relative to the cluster center,
  rotated along the local tangent (with an optional global extra rotation).

This module also runs a small self-test when executed as a script, producing demo
PNGs and PDFs for 1,2,3,4,5-set cases in /mnt/data.
"""

from typing import Optional, Sequence, Union, Tuple, List, Dict
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
from matplotlib.colors import to_rgb


# ============================================================================
# Utility helpers (standalone; no external module dependency)
# ============================================================================

def _disjoint_region_masks(masks_list: Sequence[np.ndarray]) -> dict[Tuple[int, ...], np.ndarray]:
    """
    Given a list/sequence of boolean membership masks for N sets (each shaped HxW),
    return a dict mapping every binary tuple key of length N (e.g., (1,0,1,0))
    to the corresponding disjoint region mask.
    """
    memb = np.stack(masks_list, axis=-1).astype(bool)  # (H,W,N)
    keys = list(itertools.product((0, 1), repeat=memb.shape[-1]))  # all 2^N keys
    key_arr = np.array(keys, dtype=bool)               # (K,N)

    # Compare each pixel's membership vector to every key -> (H,W,K,N), then AND over N -> (H,W,K)
    maskK = (memb[..., None, :] == key_arr[None, None, :, :]).all(axis=-1)
    return {tuple(map(int, k)): maskK[..., i] for i, k in enumerate(keys)}


def _centroid(mask: np.ndarray, X: np.ndarray, Y: np.ndarray) -> Optional[Tuple[float, float]]:
    """Centroid of `True` pixels in `mask`, mapped to coordinates via (X,Y)."""
    if not mask.any():
        return None
    yy, xx = np.where(mask)
    return (X[yy, xx].mean(), Y[yy, xx].mean())


def _rgb(color: Union[str, tuple]) -> np.ndarray:
    """Convert any Matplotlib color into an RGB float array in [0,1]."""
    return np.array(to_rgb(color), float)


def _ellipse_mask(
    X: np.ndarray, Y: np.ndarray,
    center_x: float, center_y: float,
    radius_x: float, radius_y: float,
    angle_deg: float,
) -> np.ndarray:
    """Boolean mask for a rotated ellipse."""
    theta = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    x = X - float(center_x)
    y = Y - float(center_y)
    xr =  x * cos_t + y * sin_t
    yr = -x * sin_t + y * cos_t
    return (xr / float(radius_x)) ** 2 + (yr / float(radius_y)) ** 2 <= 1.0


def _ellipse_field(
    X: np.ndarray, Y: np.ndarray,
    center_x: float, center_y: float,
    radius_x: float, radius_y: float,
    angle_deg: float,
) -> np.ndarray:
    """
    Continuous implicit field value S(x,y) for a rotated ellipse:
      S = (xr/rx)^2 + (yr/ry)^2
    Boundary is S ~= 1.
    """
    theta = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    x = X - float(center_x)
    y = Y - float(center_y)
    xr =  x * cos_t + y * sin_t
    yr = -x * sin_t + y * cos_t
    return (xr / float(radius_x)) ** 2 + (yr / float(radius_y)) ** 2


def _rotated_envelope(rx: float, ry: float, angle_deg: float) -> Tuple[float, float]:
    """Axis-aligned half-width/height that fully contains a rotated ellipse."""
    th = np.deg2rad(angle_deg)
    c, s = np.cos(th), np.sin(th)
    wx = np.sqrt((rx * c) ** 2 + (ry * s) ** 2)
    wy = np.sqrt((rx * s) ** 2 + (ry * c) ** 2)
    return wx, wy


def _binary_erode(mask: np.ndarray) -> np.ndarray:
    """
    Fast 3x3 binary erosion without external deps.
    Erosion keeps a pixel only if all 8 neighbors (and itself) are True.
    """
    h, w = mask.shape
    pad = np.pad(mask, 1, mode='constant', constant_values=False)
    eroded = np.ones_like(mask, dtype=bool)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            eroded &= pad[dy:dy + h, dx:dx + w]
    return eroded


def _normalize_angle_90(deg: float) -> float:
    """Map any angle (deg) to the equivalent angle in [-90, +90]."""
    a = float(deg)
    while a > 95.0:
        a -= 180.0
    while a < -85.0:
        a += 180.0
    return a


def _cluster_points(points: np.ndarray, radius: float) -> List[np.ndarray]:
    """
    Simple agglomerative clustering with a fixed radius.
    Returns list of cluster centers (means).
    """
    if len(points) == 0:
        return []
    pts = points.copy()
    centers: List[np.ndarray] = []
    used = np.zeros(len(pts), dtype=bool)
    for i in range(len(pts)):
        if used[i]:
            continue
        ref = pts[i]
        d = np.linalg.norm(pts - ref, axis=1)
        group_idx = (d <= radius)
        used |= group_idx
        centers.append(pts[group_idx].mean(axis=0))
    return centers


# ============================================================================
# Geometry helpers (preserve the original layouts)
# ============================================================================

def _geom1(
    sample_res: int = 600,
):
    """One circle geometry with label above and complement below."""
    r = 2.0
    pad = 0.1

    # Bounds & sampling grid
    xmin, xmax = -r - pad, r + pad
    ymin, ymax = -r - pad, r + pad
    nx = int(sample_res); ny = int(sample_res)
    xs = np.linspace(xmin, xmax, nx); ys = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(xs, ys)

    in_A = _ellipse_mask(X, Y, 0.0, 0.0, r, r, 0.0)

    label_pos = [(0.0, 1.22 * r, 0.0)]
    complement_pos = (0.0, -1.30 * r)

    return {
        "centers": [(0.0, 0.0)],
        "radii":   [(r, r)],
        "angles":  [0.0],
        "X": X, "Y": Y,
        "membership": [in_A],
        "label_positions": label_pos,
        "label_rotations": [0.0],
        "complement_pos": complement_pos,
        "limits": (xmin, xmax, ymin, ymax),
        "region_offsets": {},
        "size_unit": r,
    }


def _geom2(sample_res: int = 600, spacing_ratio: float = 1.0):
    """Two-circle geometry with ratio of AB/A = `ratio`"""
    r = 2.0
    pad = 0.1
    d = 2*r/(spacing_ratio+1)
    cxL, cxR, cy = - d/2.0, d/2.0, 0.0

    # Bounds & grid
    xmin, xmax = min(cxL, cxR) - r - pad, max(cxL, cxR) + r + pad
    ymin, ymax = cy - r - pad, cy + r + pad
    nx = int(sample_res); ny = int(sample_res)
    xs = np.linspace(xmin, xmax, nx); ys = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(xs, ys)

    in_A = _ellipse_mask(X, Y, cxL, cy, r, r, 0.0)
    in_B = _ellipse_mask(X, Y, cxR, cy, r, r, 0.0)

    label_pos = [
        (cxL, cy + 1.22 * r, 0.0),
        (cxR, cy + 1.22 * r, 0.0),
    ]
    complement_pos = (0.0, cy - 1.35 * r)
    
    region_offsets = {}
    region_offsets[(1, 0)] = (-(2*r-d)/6, 0.0) # A-only
    region_offsets[(0, 1)] = ((2*r-d)/6, 0.0) # B-only

    return {
        "centers": [(cxL, cy), (cxR, cy)],
        "radii":   [(r, r), (r, r)],
        "angles":  [0.0, 0.0],
        "X": X, "Y": Y,
        "membership": [in_A, in_B],
        "label_positions": label_pos,
        "label_rotations": [0.0, 0.0],
        "complement_pos": complement_pos,
        "limits": (xmin, xmax, ymin, ymax),
        "region_offsets": region_offsets,
        "size_unit": r,
    }


def _geom3(sample_res: int = 800, spacing: float = 1.12):
    """Three circles at the vertices of an equilateral triangle (centroid at origin)."""
    r = 2.0
    s = float(spacing) * r

    # Centers
    cxA, cyA = (0.0,  s / np.sqrt(3.0))
    cxB, cyB = (-s / 2.0, -s / (2.0 * np.sqrt(3.0)))
    cxC, cyC = ( s / 2.0, -s / (2.0 * np.sqrt(3.0)))

    # Bounds & grid
    pad  = 0.2 * r
    xmin = min(cxA, cxB, cxC) - r - pad
    xmax = max(cxA, cxB, cxC) + r + pad
    ymin = min(cyA, cyB, cyC) - r - pad
    ymax = max(cyA, cyB, cyC) + r + 2 * pad
    nx = int(sample_res); ny = int(sample_res)
    xs = np.linspace(xmin, xmax, nx); ys = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(xs, ys)

    in_A = _ellipse_mask(X, Y, cxA, cyA, r, r, 0.0)
    in_B = _ellipse_mask(X, Y, cxB, cyB, r, r, 0.0)
    in_C = _ellipse_mask(X, Y, cxC, cyC, r, r, 0.0)

    # Label positions: push outward from triangle centroid
    A = np.array([cxA, cyA]); B = np.array([cxB, cyB]); C = np.array([cxC, cyC])
    grand = (A + B + C) / 3.0

    def _out(p, k=1.22):
        v = p - grand
        n = np.linalg.norm(v)
        u = v / n if n > 1e-9 else np.array([0.0, 1.0])
        return p + u * (k * r)

    posA, posB, posC = tuple(_out(A)), tuple(_out(B)), tuple(_out(C))
    label_positions = [
        (posA[0], posA[1],   0.0),
        (posB[0], posB[1], -60.0),
        (posC[0], posC[1],  60.0),
    ]
    complement_pos = (0.0, ymin - 0.05 * r)

    return {
        "centers": [(cxA, cyA), (cxB, cyB), (cxC, cyC)],
        "radii":   [(r, r)] * 3,
        "angles":  [0.0, 0.0, 0.0],
        "X": X, "Y": Y,
        "membership": [in_A, in_B, in_C],
        "label_positions": label_positions,
        "complement_pos": complement_pos,
        "limits": (xmin, xmax, ymin, ymax),
        "region_offsets": {},
        "size_unit": r,
    }

def _geom4(sample_res: int = 900, spacing : float = 5.6):
    """Four rotated ellipses arranged in two angled pairs (+θ and −θ)."""
    # Sizes & angles
    ratio_w_to_h = 0.66 # width:height  (ry = rx / ratio_w_to_h)
    theta = 50.0        # pair1 uses +θ, pair2 uses −θ
    spacing = 5.6       # center-to-center distance (both pairs)
    pair_shift = 2.9    # midpoint shift magnitude (perpendicular +90°)
    rx = 8.0
    ry = rx / ratio_w_to_h
    
    # ---- label & complement placement knobs (SYMMETRIC) ----
    top_perp_offset = 1.4        # *units of max(rx,ry)* along +90° from top ellipses
    top_lateral_offset = -0.6    # +/- along x for left/right top labels (symmetric)
    bottom_radial_offset = 0.64  # *units of max(rx,ry)* away from grand center
    bottom_tangent_offset = 0.2  # +/- along x for left/right bottom labels
    complement_offset = 0.1      # *units of max(rx,ry)* below the lowest ellipse
    
    # ---- per-region manual nudges (for tricky shapes) ----
    a_only_offset = (-0.10, 0.2)  # A-only
    d_only_offset = (+0.10, 0.2)  # D-only

    # Unit vectors
    unit_pos = np.array([np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))], float)
    unit_neg = np.array([np.cos(np.deg2rad(-theta)), np.sin(np.deg2rad(-theta))], float)
    unit_pos_perp = np.array([-unit_pos[1], unit_pos[0]], float)  # +90°
    unit_neg_perp = np.array([-unit_neg[1], unit_neg[0]], float)

    diag_center = np.array([0.0, 0.0], float)

    # Pair +θ (A,B)
    pair1_mid = diag_center + pair_shift * unit_pos_perp
    cA = pair1_mid - 0.5 * spacing * unit_pos
    cB = pair1_mid + 0.5 * spacing * unit_pos

    # Pair −θ (C,D)
    pair2_mid = diag_center + pair_shift * unit_neg_perp
    cC = pair2_mid - 0.5 * spacing * unit_neg
    cD = pair2_mid + 0.5 * spacing * unit_neg

    # Bounds via rotated envelopes
    envs = []
    for (cx, cy, ang) in [
        (cA[0], cA[1], theta),
        (cB[0], cB[1], theta),
        (cC[0], cC[1], -theta),
        (cD[0], cD[1], -theta),
    ]:
        wx, wy = _rotated_envelope(rx, ry, ang)
        envs.append((cx - wx, cx + wx, cy - wy, cy + wy))
    xmin = min(e[0] for e in envs)
    xmax = max(e[1] for e in envs)
    ymin = min(e[2] for e in envs)
    ymax = max(e[3] for e in envs)

    size_unit = max(rx, ry)

    # Sampling grid
    nx = int(sample_res); ny = int(sample_res)
    xs = np.linspace(xmin, xmax, nx); ys = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(xs, ys)

    # Membership masks
    in_A = _ellipse_mask(X, Y, cA[0], cA[1], rx, ry, theta)
    in_B = _ellipse_mask(X, Y, cB[0], cB[1], rx, ry, theta)
    in_C = _ellipse_mask(X, Y, cC[0], cC[1], rx, ry, -theta)
    in_D = _ellipse_mask(X, Y, cD[0], cD[1], rx, ry, -theta)

    # Label placement (same logic as before)
    grand_center = np.mean(np.vstack([cA, cB, cC, cD]), axis=0)
    ellipse_info = [(cA, theta, 0), (cB, theta, 1), (cC, -theta, 2), (cD, -theta, 3)]
    top = [info for info in ellipse_info if info[0][1] >= grand_center[1]]
    bottom = [info for info in ellipse_info if info[0][1] < grand_center[1]]
    top.sort(key=lambda t: t[0][0])
    bottom.sort(key=lambda t: t[0][0])

    label_positions = [None, None, None, None]
    label_rotations = [0.0, 0.0, 0.0, 0.0]

    # Top labels: unrotated
    for idx, (cvec, ang, class_idx) in enumerate(top):
        unit_ang = np.array([np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))])
        unit_perp = np.array([-unit_ang[1], unit_ang[0]])
        direction_perp = unit_perp if np.dot(unit_perp, cvec - grand_center) >= 0 else -unit_perp
        base_xy = cvec + direction_perp * (float(top_perp_offset) * size_unit)
        lateral = (-1 if idx == 0 else 1) * float(top_lateral_offset) * size_unit
        label_xy = (base_xy[0] + lateral, base_xy[1])
        label_positions[class_idx] = label_xy
        label_rotations[class_idx] = 0.0

    # Bottom labels: rotated; LEFT = −angle, RIGHT = +angle
    for idx, (cvec, ang, class_idx) in enumerate(bottom):
        from_center = cvec - grand_center
        n = np.linalg.norm(from_center)
        u = from_center / n if n > 1e-9 else np.array([0.0, -1.0])
        base_xy = cvec + u * (float(bottom_radial_offset) * size_unit)
        lateral = (-1 if idx == 0 else 1) * float(bottom_tangent_offset) * size_unit
        label_xy = (base_xy[0] + lateral, base_xy[1])
        label_positions[class_idx] = label_xy
        label_rotations[class_idx] = -theta if idx == 0 else theta
        
    # Complement below
    lowest_y = min(cA[1] - ry, cB[1] - ry, cC[1] - ry, cD[1] - ry)
    complement_pos = (diag_center[0], lowest_y - (float(complement_offset) * size_unit))

    # Region-specific nudges
    region_offsets: Dict[Tuple[int, int, int, int], Tuple[float, float]] = {}
    dx_A, dy_A = a_only_offset
    dx_D, dy_D = d_only_offset
    region_offsets[(1, 0, 0, 0)] = (dx_A * size_unit, dy_A * size_unit)  # A-only
    region_offsets[(0, 0, 0, 1)] = (dx_D * size_unit, dy_D * size_unit)  # D-only

    # Pack label positions with rotations
    label_positions_with_rot = [(xy[0], xy[1], rot) for xy, rot in zip(label_positions, label_rotations)]

    return {
        "centers": [(cA[0], cA[1]), (cB[0], cB[1]), (cC[0], cC[1]), (cD[0], cD[1])],
        "radii":   [(rx, ry)] * 4,
        "angles":  [theta, theta, -theta, -theta],
        "X": X, "Y": Y,
        "membership": [in_A, in_B, in_C, in_D],
        "label_positions": label_positions_with_rot,
        "complement_pos": complement_pos,
        "limits": (xmin, xmax, ymin, ymax),
        "region_offsets": region_offsets,
        "size_unit": size_unit,
    }


def _geom5(sample_res: int = 900):
    """
    Five ellipses at angles base-72°*k, k=0..4 (clockwise ordering: A,B,C,D,E).
    """
    # Sizes & angles
    ratio_w_to_h = 1.6          # width:height (ry = rx / ratio_w_to_h)
    base_angle_deg = 90         # global rotation offset for the 5 spokes
    trans_along = 1.0           # translation along ellipse's own angle
    trans_perp = 0.5            # translation perpendicular to its angle
    
    # ---- Class label placement (support-point method) ----
    label_gap_units = 0.02
    label_tangent_units = 0.2
    label_rotation_extra = -26.5

    rx = 6.0
    ry = rx / ratio_w_to_h

    base = float(base_angle_deg)
    # Clockwise order: decrease by 72.0° each step
    angles = [base - 72.0 * k for k in range(5)]
    centers: List[Tuple[float, float]] = []

    root_center = np.array([0.0, 0.0], float)

    # Compute centers
    for ang in angles:
        th = np.deg2rad(ang)
        u = np.array([np.cos(th), np.sin(th)], float)
        u_perp = np.array([-u[1], u[0]], float)  # +90°
        c = root_center + float(trans_along) * u + float(trans_perp) * u_perp
        centers.append((float(c[0]), float(c[1])))

    # Bounds via rotated envelopes
    envs = []
    for (cx, cy), ang in zip(centers, angles):
        wx, wy = _rotated_envelope(rx, ry, ang)
        envs.append((cx - wx, cx + wx, cy - wy, cy + wy))
    xmin = min(e[0] for e in envs)
    xmax = max(e[1] for e in envs)
    ymin = min(e[2] for e in envs)
    ymax = max(e[3] for e in envs)

    size_unit = max(rx, ry)

    # Sampling grid
    nx = int(sample_res); ny = int(sample_res)
    xs = np.linspace(xmin, xmax, nx); ys = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(xs, ys)

    # Membership masks
    memberships = []
    for (cx, cy), ang in zip(centers, angles):
        memberships.append(_ellipse_mask(X, Y, cx, cy, rx, ry, ang))

    # ------ Label placement using ellipse support function ------
    def _rot(vx, vy, ang_deg):
        th = np.deg2rad(ang_deg)
        c, s = np.cos(th), np.sin(th)
        return np.array([c * vx - s * vy, s * vx + c * vy], float)

    grand_center = np.mean(np.array(centers), axis=0)
    label_positions: List[Tuple[float, float, float]] = []

    for (cx, cy), ang in zip(centers, angles):
        c = np.array([cx, cy], float)

        # 1) outward direction from cluster center
        d = c - grand_center
        n_d = np.linalg.norm(d)
        if n_d < 1e-9:
            d = _rot(1.0, 0.0, ang)  # fallback along ellipse direction
        else:
            d = d / n_d

        # 2) transform to ellipse local frame (axis-aligned)
        d_local = _rot(d[0], d[1], -ang)

        # 3) support point on axis-aligned ellipse in direction d_local
        denom = np.sqrt((rx * d_local[0])**2 + (ry * d_local[1])**2)
        if denom < 1.0e-12:
            # degenerate; pick rightmost point
            p_local = np.array([rx, 0.0], float)
        else:
            p_local = np.array([ (rx**2) * d_local[0] / denom,
                                 (ry**2) * d_local[1] / denom ], float)

        # 4) back to world coords
        p_world = _rot(p_local[0], p_local[1], ang) + c

        # 5) outward normal at support point via gradient in local frame
        grad_local = np.array([2.0 * p_local[0] / (rx**2), 2.0 * p_local[1] / (ry**2)], float)
        n_world = _rot(grad_local[0], grad_local[1], ang)
        n_norm = np.linalg.norm(n_world)
        n_world = n_world / n_norm if n_norm > 1.0e-12 else d  # fallback

        # 6) tangent (rotate normal by -90° for clockwise consistency)
        t_world = np.array([+n_world[1], -n_world[0]], float)

        # 7) place label: gap outward + small tangential nudge
        label_pos = p_world + float(label_gap_units) * size_unit * n_world + float(label_tangent_units) * size_unit * t_world

        # 8) rotation along tangent (+ extra)
        rot = float(np.degrees(np.arctan2(t_world[1], t_world[0])) + float(label_rotation_extra))

        # 9) micro-adjust to ensure outside (increase gap if needed)
        #    Check ellipse implicit equation at label point in local frame.
        lp_local = _rot(label_pos[0] - cx, label_pos[1] - cy, -ang)
        val = (lp_local[0] / rx) ** 2 + (lp_local[1] / ry) ** 2
        tries = 0
        gap = float(label_gap_units) * size_unit
        while val <= 1.02 and tries < 4:  # small margin beyond boundary
            gap *= 1.25
            label_pos = p_world + gap * n_world + float(label_tangent_units) * size_unit * t_world
            lp_local = _rot(label_pos[0] - cx, label_pos[1] - cy, -ang)
            val = (lp_local[0] / rx) ** 2 + (lp_local[1] / ry) ** 2
            tries += 1

        label_positions.append((float(label_pos[0]), float(label_pos[1]), rot))

    # Complement below
    lowest_y = min(cy - ry for (cx, cy) in centers)
    complement_pos = (0.15 * size_unit, lowest_y - (0.4 * size_unit))

    return {
        "centers": centers,
        "radii":   [(rx, ry)] * 5,
        "angles":  angles,
        "X": X, "Y": Y,
        "membership": memberships,
        "label_positions": label_positions,
        "complement_pos": complement_pos,
        "limits": (xmin, xmax, ymin, ymax),
        "region_offsets": {},              # no manual nudges by default
        "size_unit": size_unit
    }


# ============================================================================
# Generic renderer
# ============================================================================

def venn(
    values,
    class_names: Sequence[str],
    colors: Optional[Sequence[Union[str, tuple]]] = None,
    title: Optional[str] = None,
    outfile: Optional[str] = None,
    dpi: int = 100,
    rotate_region_labels: Optional[bool] = None,
    **kwargs,
) -> Optional[Figure]:
    """
    Unified Venn plotter for N in {1,2,3,4,5}.

    Parameters
    ----------
    values : array-like with shape (2,)*N
        Truth-table order values. 0=absent,1=present per axis.
        For N=1: [outside, A]
        For N=2: [[00,01],[10,11]], etc.
    class_names : list[str]
        Names of the sets (length N).
    colors : list[str|tuple], optional
        Colors for each set. Defaults to Matplotlib prop cycle.
    title : optional
        Plot title
    outfile : optional
        Optional output path. If `outfile` is given, the figure
        is saved and the function returns None.
    dpi : int
        DPI for saving the figure (if `outfile` is given).
    rotate_region_labels : bool
        If True, compute a per-region “ideal direction” from detected
        corners (3 -> longest edge; 4 -> longer of lines connecting side midpoints,
        with 1% tie → longest diagonal) and rotate the region label accordingly.
    kwargs : forwarded to the corresponding geometry helper
    """
    # ---- Determine N and build geometry ----
    arr = np.asarray(values, dtype=object)

    if arr.ndim == 1:
        N = 1
        if arr.shape != (2,):
            raise ValueError("For N=1, values must have shape (2,) as [0, 1].")
        geom = _geom1(**{k: v for k, v in kwargs.items() if k in {"radius", "center_xy", "sample_res"}})

    elif arr.ndim == 2:
        N = 2
        if arr.shape != (2, 2):
            raise ValueError("For N=2, values must be 2x2.")
        geom = _geom2(**{k: v for k, v in kwargs.items() if k in {"radius", "d_factor", "sample_res"}})

    elif arr.ndim == 3:
        N = 3
        if arr.shape != (2, 2, 2):
            raise ValueError("For N=3, values must be 2x2x2.")
        geom = _geom3(**{k: v for k, v in kwargs.items() if k in {"radius", "pair_sep_factor", "sample_res"}})
        if rotate_region_labels is None:
            rotate_region_labels = False

    elif arr.ndim == 4:
        N = 4
        if arr.shape != (2, 2, 2, 2):
            raise ValueError("For N=4, values must be 2x2x2x2.")
        geom = _geom4(**kwargs)

    elif arr.ndim == 5:
        N = 5
        if arr.shape != (2, 2, 2, 2, 2):
            raise ValueError("For N=5, values must be 2x2x2x2x2.")
        geom = _geom5(**kwargs)

    else:
        raise ValueError("Only N in {1,2,3,4,5} are supported.")

    if len(class_names) != N:
        raise ValueError(f"class_names must have length {N}")

    # ---- Colors ----
    if colors is None:
        cycle = plt.rcParams["axes.prop_cycle"].by_key().get(
            "color",
            ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
        )
        colors = [cycle[i % len(cycle)] for i in range(N)]
    rgbs = list(map(_rgb, colors))

    # ---- Rasterize membership to image (RGBA) ----
    X, Y = geom["X"], geom["Y"]
    membership = [m.astype(float) for m in geom["membership"]]  # list (H,W)
    membership_stack = np.stack(membership, axis=-1)             # (H,W,N)
    count = membership_stack.sum(axis=-1, keepdims=True)         # (H,W,1)
    rgb_sum = (membership_stack @ np.stack(rgbs, axis=0))        # (H,W,3)
    with np.errstate(invalid="ignore", divide="ignore"):
        rgb_avg = np.where(count > 0, rgb_sum / count, 0.0)
    alpha = (count[..., 0] > 0).astype(float)
    rgba = np.dstack([rgb_avg, alpha])

    # ---- Figure ----
    fig, ax = plt.subplots(figsize=(9.6, 8.6))
    xmin, xmax, ymin, ymax = geom["limits"]
    ax.imshow(rgba, origin="lower", extent=[xmin, xmax, ymin, ymax], interpolation="none", zorder=2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.margins(0.0, 0.0)

    # ---- Outlines (two-pass to hide raster seams) ----
    outline_lw = 2.0

    # Pass 1: fully opaque outlines beneath (alpha=1) to mask seams
    for (cx, cy), (rx, ry), ang, col in zip(geom["centers"], geom["radii"], geom["angles"], rgbs):
        ax.add_patch(
            Ellipse(
                (cx, cy), 2 * rx, 2 * ry, angle=ang,
                fill=False, lw=outline_lw,
                edgecolor=(col[0], col[1], col[2], 1.0),
                zorder=4.9
            )
        )

    # Pass 2: outlines with global `alpha` 0.5 on top
    for (cx, cy), (rx, ry), ang, col in zip(geom["centers"], geom["radii"], geom["angles"], rgbs):
        ax.add_patch(
            Ellipse(
                (cx, cy), 2 * rx, 2 * ry, angle=ang,
                fill=False, lw=outline_lw,
                edgecolor=(col[0], col[1], col[2], 0.5),
                zorder=5.0
            )
        )

    # ---- Per-ellipse continuous fields (for corner detection) ----
    fields = []
    for (cx, cy), (rx, ry), ang in zip(geom["centers"], geom["radii"], geom["angles"]):
        fields.append(_ellipse_field(X, Y, cx, cy, rx, ry, ang))
    fields = np.stack(fields, axis=-1)  # (H,W,N)

    # ---- Region values (per disjoint area) ----
    masks = _disjoint_region_masks([m.astype(bool) for m in membership])
    size_unit = float(geom["size_unit"])
    region_offsets = geom.get("region_offsets", {})

    # Compute ideal rotations per region (if enabled)
    region_rotations: Dict[Tuple[int, ...], float] = {}

    if rotate_region_labels is not False:
        eps = 0.02  # tolerance for |S-1| near ellipse boundary
        corner_cluster_radius = 0.03 * size_unit  # cluster radius in world coords

        for key, mask in masks.items():
            if not any(key):
                continue
            if not mask.any():
                continue

            # Region boundary
            eroded = _binary_erode(mask)
            boundary = mask & (~eroded)
            by, bx = np.where(boundary)
            if by.size < 6:
                continue  # too small to be robust

            # World coords of boundary points
            bx_world = X[by, bx]
            by_world = Y[by, bx]
            pts = np.column_stack((bx_world, by_world))

            # Corner candidates: points near intersection of >=2 ellipse boundaries
            fvals = fields[by, bx, :]  # (B,N)
            near = np.abs(fvals - 1.0) < eps
            multi_near = near.sum(axis=1) >= 2
            corner_candidates = pts[multi_near]

            # Cluster to get distinct corners
            corners = _cluster_points(corner_candidates, radius=corner_cluster_radius)

            # Only act on triangles / quads
            if len(corners) == 3:
                # --- FIX: if all three sides within 1%, pick the one nearest 0°;
                #          else if two sides within 1%, select the third side; otherwise longest. ---
                C = np.array(corners)
                e01 = C[1] - C[0]; l01 = np.linalg.norm(e01)
                e12 = C[2] - C[1]; l12 = np.linalg.norm(e12)
                e02 = C[2] - C[0]; l02 = np.linalg.norm(e02)

                def _near(a, b):
                    m = max(a, b)
                    return m > 0 and abs(a - b) <= 0.01 * m

                # All sides near-equal?
                if _near(l01, l12) and _near(l12, l02):
                    candidates = [e01, e12, e02]
                    def _ang_abs(v):
                        return abs(_normalize_angle_90(np.degrees(np.arctan2(v[1], v[0]))))
                    chosen_vec = min(candidates, key=_ang_abs)
                else:
                    # Exactly-two-near-equal → choose the third side
                    if _near(l01, l12):
                        chosen_vec = e02  # third side (0,2)
                    elif _near(l12, l02):
                        chosen_vec = e01  # third side (0,1)
                    elif _near(l01, l02):
                        chosen_vec = e12  # third side (1,2)
                    else:
                        # Fallback: longest side
                        if l01 >= l12 and l01 >= l02:
                            chosen_vec = e01
                        elif l12 >= l01 and l12 >= l02:
                            chosen_vec = e12
                        else:
                            chosen_vec = e02

                v = chosen_vec
                ang = np.degrees(np.arctan2(v[1], v[0]))
                region_rotations[key] = _normalize_angle_90(ang)

            elif len(corners) == 4:
                # Order corners around their centroid to define sides
                C = np.array(corners)
                gc = C.mean(axis=0)
                angles = np.arctan2(C[:, 1] - gc[1], C[:, 0] - gc[0])
                order = np.argsort(angles)
                C = C[order]  # A,B,C,D around

                # Opposite side midpoints
                m1 = 0.5 * (C[0] + C[1]); m3 = 0.5 * (C[2] + C[3])  # AB vs CD
                m2 = 0.5 * (C[1] + C[2]); m4 = 0.5 * (C[3] + C[0])  # BC vs DA

                v13 = m3 - m1
                v24 = m4 - m2
                len13 = np.linalg.norm(v13)
                len24 = np.linalg.norm(v24)

                # If within 1% → use the longest diagonal instead
                if max(len13, len24) > 0 and abs(len13 - len24) <= 0.01 * max(len13, len24):
                    d02 = C[2] - C[0]  # diagonal AC
                    d13 = C[3] - C[1]  # diagonal BD
                    if np.linalg.norm(d02) >= np.linalg.norm(d13):
                        v = d02
                    else:
                        v = d13
                else:
                    v = v13 if len13 >= len24 else v24

                ang = np.degrees(np.arctan2(v[1], v[0]))
                region_rotations[key] = _normalize_angle_90(ang)
            # Other counts: leave rotation at default (0°)

    # ---- Region label drawing (with optional downward shift after rotation) ----
    region_fontsize = 14  # keep consistent with previous code
    # Convert text height (points) -> pixels -> data units (along vertical)
    def _data_units_for_pixels(ax, px: float) -> float:
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        p_disp = ax.transData.transform((cx, cy))
        p2_disp = (p_disp[0], p_disp[1] - px)
        p2_data = ax.transData.inverted().transform(p2_disp)
        return abs(p2_data[1] - cy)

    text_height_pts = float(region_fontsize)  # approximate
    text_height_px = text_height_pts * fig.dpi / 72.0
    down_len_data = 0.1 * _data_units_for_pixels(ax, text_height_px)

    # Draw region texts (with optional rotation + post-rotation downward shift)
    for key, mask in masks.items():
        if not any(key):
            continue  # skip all-zeros region here (handled as complement below)
        value = arr[key]
        if value is None:
            continue
        pos = _centroid(mask, X, Y)
        if pos is None:
            continue
        dx, dy = region_offsets.get(tuple(map(int, key)), (0.0, 0.0))
        rot_val = float(region_rotations.get(key, 0.0))

        # Compute “down” direction in world coords for given rotation:
        # baseline unit b = [cosθ, sinθ]; upward normal u = [-sinθ, cosθ]; downward = -u
        theta = np.deg2rad(rot_val)
        down_vec = np.array([np.sin(theta), -np.cos(theta)], float) * down_len_data

        ax.text(
            pos[0] + dx + down_vec[0], pos[1] + dy + down_vec[1], f"{value}",
            ha="center", va="center", fontsize=region_fontsize, zorder=8,
            rotation=rot_val, rotation_mode="anchor"
        )

    # ---- Complement (all-zeros) ----
    zeros = (0,) * N
    if arr[zeros] is not None:
        cx, cy = geom["complement_pos"]
        ax.text(cx, cy, f"{arr[zeros]}", ha="center", va="center", fontsize=14)

    # ---- Class labels ----
    for (x, y, rot), name, col in zip(geom["label_positions"], class_names, rgbs):
        ax.text(x, y, name, ha="center", va="center", fontsize=16,
                color=tuple(col), rotation=rot, rotation_mode="anchor")

    # ---- Expand limits to include labels & complement ----
    limit_pad_units = 0.1
    label_pts = [(xy[0], xy[1]) for xy in geom["label_positions"]]
    extras = []
    if arr[zeros] is not None:
        extras.append(geom["complement_pos"])
    if label_pts or extras:
        pts = np.array(label_pts + extras)
        min_x = min(xmin, np.min(pts[:, 0]))
        max_x = max(xmax, np.max(pts[:, 0]))
        min_y = min(ymin, np.min(pts[:, 1]))
        max_y = max(ymax, np.max(pts[:, 1]))
        pad_abs = limit_pad_units * size_unit
        ax.set_xlim(min_x - pad_abs, max_x + pad_abs)
        ax.set_ylim(min_y - pad_abs, max_y + pad_abs)

    # ---- Title / export ----
    if title:
        ax.set_title(title, fontsize=20)

    if outfile:
        fig.savefig(outfile, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return None

    return fig

