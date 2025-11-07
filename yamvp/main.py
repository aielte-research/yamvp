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

from typing import Optional, Sequence, Union, Tuple, List, Dict, Callable
import itertools
import numbers
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
from matplotlib.colors import to_rgb
from scipy.ndimage import distance_transform_edt, binary_erosion
from scipy.cluster.hierarchy import fclusterdata

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

def _visual_center(mask: np.ndarray, X: np.ndarray, Y: np.ndarray):
    """
    Visual center via Euclidean distance transform (SciPy).
    """
    if not mask.any():
        return None
    dist = distance_transform_edt(mask)
    yy, xx = np.unravel_index(np.argmax(dist), dist.shape)
    return float(X[yy, xx]), float(Y[yy, xx])

def _rgb(color: Union[str, tuple]) -> np.ndarray:
    """Convert any Matplotlib color into an RGB float array in [0,1]."""
    return np.array(to_rgb(color), float)


def _color_mix_average(colors: Sequence[np.ndarray]) -> np.ndarray:
    """
    Default color mixing: simple average of the provided RGB colors.
    """
    if not colors:
        return np.zeros(3, float)
    arr = np.stack([np.array(c, float) for c in colors], axis=0)
    return arr.mean(axis=0)


def _color_mix_alpha_stack(colors: Sequence[np.ndarray], alpha: float = 0.5) -> np.ndarray:
    """
    Mix colors by stacking them with a fixed per-layer alpha.
    """
    if not colors:
        return np.zeros(3, float)
    a = float(alpha)
    a = max(0.0, min(1.0, a))
    c = np.array(colors[0], float)
    for col in colors[1:]:
        col_arr = np.array(col, float)
        c = c * (1.0 - a) + col_arr * a
    return c

def _color_mix_subtractive(colors):
    """
    Mix colors by subtracive mixing.
    """
    if not colors:
        return np.zeros(3, float)
    arr = np.stack([np.array(c, float) for c in colors], axis=0)
    return np.abs(1.0 - (np.prod(1-arr, axis=0))*(len(colors)**0.25))

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
    x = X - float(center_x)
    y = Y - float(center_y)
    xr =  x * np.cos(theta) + y * np.sin(theta)
    yr = -x * np.sin(theta) + y * np.cos(theta)
    return (xr / float(radius_x)) ** 2 + (yr / float(radius_y)) ** 2

def _ellipse_mask(
    X: np.ndarray, Y: np.ndarray,
    center_x: float, center_y: float,
    radius_x: float, radius_y: float,
    angle_deg: float,
) -> np.ndarray:
    """Boolean mask for a rotated ellipse."""
    return _ellipse_field(X, Y, center_x, center_y, radius_x, radius_y, angle_deg) <= 1.0

def _rotated_envelope(rx: float, ry: float, angle_deg: float) -> Tuple[float, float]:
    """Axis-aligned half-width/height that fully contains a rotated ellipse."""
    th = np.deg2rad(angle_deg)
    c, s = np.cos(th), np.sin(th)
    wx = np.sqrt((rx * c) ** 2 + (ry * s) ** 2)
    wy = np.sqrt((rx * s) ** 2 + (ry * c) ** 2)
    return wx, wy

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
    Cluster points with a fixed radius using SciPy, return cluster centers.
    """
    if len(points) == 0:
        return []
    labels = fclusterdata(points, t=radius, criterion="distance")
    centers = [points[labels == lab].mean(axis=0) for lab in np.unique(labels)]
    return centers

def _circle_intersection_area(r1: float, r2: float, d: float) -> float:
    """
    Area of intersection of two circles of radii r1, r2 separated by distance d.
    """
    r1 = float(r1)
    r2 = float(r2)
    d = float(d)
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return float(np.pi * min(r1, r2) ** 2)

    x = (d * d + r1 * r1 - r2 * r2) / (2.0 * d * r1)
    y = (d * d + r2 * r2 - r1 * r1) / (2.0 * d * r2)
    x = float(np.clip(x, -1.0, 1.0))
    y = float(np.clip(y, -1.0, 1.0))
    alpha = float(np.arccos(x))
    beta = float(np.arccos(y))
    term = (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)
    term = max(0.0, term)
    inter = r1 * r1 * alpha + r2 * r2 * beta - 0.5 * float(np.sqrt(term))
    return float(inter)

def _solve_circle_distance_for_area(r1: float, r2: float, target_area: float,
                                    tol: float = 1e-6, max_iter: int = 100) -> float:
    """
    Given radii r1, r2 and desired intersection area (0 < target_area < area_small),
    find distance d between centers such that the intersection area matches target_area.
    """
    r1 = float(r1)
    r2 = float(r2)
    target_area = float(target_area)

    if target_area <= 0.0:
        return r1 + r2

    area_small = float(np.pi * min(r1, r2) ** 2)
    if target_area >= area_small:
        return 0.0

    lo = abs(r1 - r2)
    hi = r1 + r2

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        val = _circle_intersection_area(r1, r2, mid)
        if abs(val - target_area) <= tol * area_small:
            return mid
        if val > target_area:
            lo = mid
        else:
            hi = mid

    return 0.5 * (lo + hi)

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

def _geom2(
    sample_res: int = 600,
    spacing_ratio: Optional[float] = None,
    radius: Optional[float] = None,
    d_factor: Optional[float] = None,
    area_proportional: bool = False,
    values: Optional[np.ndarray] = None,
):
    """
    Two-circle geometry.

    - Default: symmetric circles with spacing controlled by `spacing_ratio` (or
      `d_factor` as a backward-compatible alias).
    - If `area_proportional` is True and `values` is a 2x2 table of non-negative
      numbers (or None), then the circle areas and their intersection are made
      proportional to the region values for N=2.
    """
    # Base radius and spacing
    r0 = 1.5 if radius is None else float(radius)

    if spacing_ratio is None:
        if d_factor is not None:
            spacing_ratio_val = float(d_factor)
        else:
            spacing_ratio_val = 1.0
    else:
        spacing_ratio_val = float(spacing_ratio)

    pad = 0.1

    # Defaults: symmetric radii and spacing
    radA = r0
    radB = r0
    d = 2.0 * r0 / (spacing_ratio_val + 1.0)
    cxL, cxR, cy = -d / 2.0, d / 2.0, 0.0

    # Flags for whether area-proportional geometry was actually used
    area_logic_ok = False
    edge_case = False

    # Optional area-proportional scaling for N=2
    if area_proportional and values is not None:
        arr = np.asarray(values, dtype=object)
        if arr.shape == (2, 2):
            flat = arr.ravel()

            # 1) All inputs numbers or None
            types_ok = True
            for v in flat:
                if v is None:
                    continue
                if not isinstance(v, numbers.Real):
                    types_ok = False
                    break

            # 2) No negatives
            if types_ok:
                for v in flat:
                    if v is None:
                        continue
                    if float(v) < 0.0:
                        types_ok = False
                        break

            if types_ok:
                v00 = arr[0, 0]
                v01 = arr[0, 1]  # B-only
                v10 = arr[1, 0]  # A-only
                v11 = arr[1, 1]  # intersection
                regs = [v01, v10, v11]

                # Require explicit numeric values for non-complement regions
                if any(v is None for v in regs):
                    types_ok = False

            if types_ok:
                # More than one zero (excluding complement) -> bail out to original geometry
                zero_count = sum(
                    1 for v in regs
                    if v is not None and float(v) == 0.0
                )
                if zero_count > 1:
                    types_ok = False

            if types_ok:
                a_only = float(v10)
                b_only = float(v01)
                both = float(v11)

                A_total = a_only + both
                B_total = b_only + both

                if A_total > 0.0 and B_total > 0.0:
                    area_logic_ok = True

                    # Scale radii so their areas reflect totals up to a global factor
                    if A_total >= B_total:
                        radA = r0
                        radB = r0 * np.sqrt(B_total / A_total)
                        total_big = A_total
                    else:
                        radB = r0
                        radA = r0 * np.sqrt(A_total / B_total)
                        total_big = B_total

                    # Global proportionality constant so that big set area ~ total_big
                    k = float(np.pi * (r0 ** 2)) / float(total_big)
                    target_intersection_area = k * both

                    # Edge cases for distance
                    if both <= 0.0:
                        # No intersection -> circles just touch externally
                        d = radA + radB
                        edge_case = True
                    elif a_only == 0.0 or b_only == 0.0:
                        # A-B == 0 or B-A == 0: internal tangency
                        # d + r_small = r_big  => d = |r_big - r_small|
                        d = abs(radA - radB)
                        edge_case = True
                    else:
                        # General case: match intersection area
                        d = _solve_circle_distance_for_area(radA, radB, target_intersection_area)
                # else: leave area_logic_ok False -> fall back to default geometry

    # If area-proportional geometry is active, shift centers using the asymmetric formula
    if area_logic_ok:
        cxL = -(d - radA + radB) / 2.0
        cxR =  (d + radA - radB) / 2.0
        # cy stays 0

    # Whether to use special outside label geometry
    use_special_labels = area_proportional and area_logic_ok and not edge_case

    # Bounds & grid (use max radius)
    r_max = max(radA, radB)
    xmin, xmax = min(cxL, cxR) - r_max - pad, max(cxL, cxR) + r_max + pad
    ymin, ymax = cy - r_max - pad, cy + r_max + pad
    nx = int(sample_res); ny = int(sample_res)
    xs = np.linspace(xmin, xmax, nx); ys = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(xs, ys)

    in_A = _ellipse_mask(X, Y, cxL, cy, radA, radA, 0.0)
    in_B = _ellipse_mask(X, Y, cxR, cy, radB, radB, 0.0)

    size_unit = r_max

    # ---- Helper: top circle–circle intersection if it exists ----
    def _top_intersection_point(cx1: float, r1: float, cx2: float, r2: float, cy_: float) -> Optional[Tuple[float, float]]:
        d_centers = abs(cx2 - cx1)
        if d_centers <= 1e-9:
            # Coincident centers -> degenerate
            return None
        if d_centers > r1 + r2:
            # Disjoint
            return None
        if d_centers < abs(r1 - r2) - 1e-9:
            # Strict containment without tangency
            return None

        a = (r1 * r1 - r2 * r2 + d_centers * d_centers) / (2.0 * d_centers)
        h_sq = r1 * r1 - a * a
        if h_sq < 0.0:
            h_sq = 0.0
        h = float(np.sqrt(h_sq))
        x = cx1 + a * np.sign(cx2 - cx1)
        y = cy_ + h
        return (x, y)

    inter = _top_intersection_point(cxL, radA, cxR, radB, cy)

    # ---- Class label positions ----
    if use_special_labels and inter is not None and radA > 0.0 and radB > 0.0:
        ix, iy = inter
        offset = 0.12 * size_unit

        # Left circle: OUTSIDE direction (towards left)
        u0L = np.array([-1.0, 0.0], float)
        v1L = np.array([ix - cxL, iy - cy], float)
        n1L = np.linalg.norm(v1L)
        if n1L > 1e-12:
            u1L = v1L / n1L
        else:
            u1L = u0L
        u_mid_L = u0L + u1L
        n_mid_L = np.linalg.norm(u_mid_L)
        if n_mid_L < 1e-12:
            u_mid_L = u0L
        else:
            u_mid_L = u_mid_L / n_mid_L

        bxL = cxL + radA * u_mid_L[0]
        byL = cy + radA * u_mid_L[1]
        lxL = bxL + offset * u_mid_L[0]
        lyL = byL + offset * u_mid_L[1]
        tL = np.array([-u_mid_L[1], u_mid_L[0]], float)
        rotL = _normalize_angle_90(float(np.degrees(np.arctan2(tL[1], tL[0]))))

        # Right circle: OUTSIDE direction (towards right)
        u0R = np.array([1.0, 0.0], float)
        v1R = np.array([ix - cxR, iy - cy], float)
        n1R = np.linalg.norm(v1R)
        if n1R > 1e-12:
            u1R = v1R / n1R
        else:
            u1R = u0R
        u_mid_R = u0R + u1R
        n_mid_R = np.linalg.norm(u_mid_R)
        if n_mid_R < 1e-12:
            u_mid_R = u0R
        else:
            u_mid_R = u_mid_R / n_mid_R

        bxR = cxR + radB * u_mid_R[0]
        byR = cy + radB * u_mid_R[1]
        lxR = bxR + offset * u_mid_R[0]
        lyR = byR + offset * u_mid_R[1]
        tR = np.array([-u_mid_R[1], u_mid_R[0]], float)
        rotR = _normalize_angle_90(float(np.degrees(np.arctan2(tR[1], tR[0]))))

        label_pos = [
            (lxL, lyL, rotL),
            (lxR, lyR, rotR),
        ]
    else:
        # Original simple label positions: above each circle, no rotation
        label_pos = [
            (cxL, cy + radA + (radA + radB)*0.08, 0.0),
            (cxR, cy + radB + (radA + radB)*0.08, 0.0),
        ]

    complement_pos = (0.0, cy - 1.35 * r_max)
    
    # No region_offsets hack for N=2; region labels handled specially in `venn`.
    region_offsets: Dict[Tuple[int, int], Tuple[float, float]] = {}

    return {
        "centers": [(cxL, cy), (cxR, cy)],
        "radii":   [(radA, radA), (radB, radB)],
        "angles":  [0.0, 0.0],
        "X": X, "Y": Y,
        "membership": [in_A, in_B],
        "label_positions": label_pos,
        "label_rotations": [0.0, 0.0],
        "complement_pos": complement_pos,
        "limits": (xmin, xmax, ymin, ymax),
        "region_offsets": region_offsets,
        "size_unit": size_unit,
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

    # For N=3, all disjoint regions use visual centers.
    region_label_method_default = "visual_center"
    region_label_method_overrides: Dict[Tuple[int, int, int], str] = {}

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
        "region_label_method_default": region_label_method_default,
        "region_label_method_overrides": region_label_method_overrides,
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

    # Region-specific label method:
    region_label_method_default = "centroid"
    region_label_method_overrides: Dict[Tuple[int, int, int, int], str] = {
        (1, 0, 0, 0): "visual_center",  # A
        (0, 1, 0, 0): "visual_center",  # B
        (0, 0, 1, 0): "visual_center",  # C
        (0, 0, 0, 1): "visual_center",  # D
        (1, 0, 0, 1): "visual_center",  # AD
    }

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
        "region_offsets": {},
        "size_unit": size_unit,
        "region_label_method_default": region_label_method_default,
        "region_label_method_overrides": region_label_method_overrides,
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
            p_local = np.array([ (rx**2) * d_local[0] / denom, (ry**2) * d_local[1] / denom ], float)

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
    
    # Region-specific nudges
    region_offsets: Dict[Tuple[int, int, int, int, int], Tuple[float, float]] = {}
    region_offsets[(0, 1, 0, 1, 1)] = ( -0.015 * size_unit, 0.0 * size_unit)  # BDE-only
    region_offsets[(0, 1, 1, 1, 0)] = ( -0.005 * size_unit, -0.005 * size_unit)  # BCD-only
    region_offsets[(1, 0, 1, 0, 1)] = ( -0.002 * size_unit, 0.005 * size_unit)  # ACE-only
    
    region_label_method_default = "centroid"
    region_label_method_overrides: Dict[Tuple[int, int, int, int, int], str] = {}

    return {
        "centers": centers,
        "radii":   [(rx, ry)] * 5,
        "angles":  angles,
        "X": X, "Y": Y,
        "membership": memberships,
        "label_positions": label_positions,
        "complement_pos": complement_pos,
        "limits": (xmin, xmax, ymin, ymax),
        "region_offsets": region_offsets,
        "size_unit": size_unit,
        "region_label_method_default": region_label_method_default,
        "region_label_method_overrides": region_label_method_overrides,
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
    color_mixing: Union[str, Callable[[Sequence[np.ndarray]], np.ndarray]] = "subtractive",
    text_color: Optional[str] = "black",
    region_label_fontsize: int = 12,
    class_name_fontsize: int = 16,
    title_fontsize: int = 20,
    area_proportional: bool = False,
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
        (Ignored for N=2, where region labels sit on y=0 as described below.)
    color_mixing : {"average", "alpha"} or callable, optional
        How to mix set colors into the region color. If a string:
          - "average": simple arithmetic mean of member colors
          - "alpha": stack colors with a fixed alpha over white background
        If a callable, it must accept a list of RGB colors (each shape (3,))
        and return a single mixed RGB color (shape (3,)).
    text_color : Optional[str]
        Color for text labels (not the class colors, those are set via `colors`).
        If None, region labels are automatically set to black or white based on
        the region's background color.
    region_label_fontsize: int
        Font size for region labels.
    class_name_fontsize: int
        Font size for class (set) names.
    title_fontsize: int
        Font size for the title.
    area_proportional : bool
        If True and N==2 with suitable numeric inputs, scale the two circles so
        that their areas (and the intersection area, via center distance) are
        proportional to the provided region values. If the inputs are not all
        numbers/None, contain negatives, or more than one non-complement region
        is 0, this flag is ignored and the original geometry is used.
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
        geom_kwargs = {k: v for k, v in kwargs.items() if k in {"radius", "d_factor", "sample_res", "spacing_ratio"}}
        geom = _geom2(area_proportional=area_proportional, values=arr, **geom_kwargs)

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
        cycle = ['#2ca02c', '#ff7f0e', '#1f77b4', '#9467bd', "#d62728"]
        colors = [cycle[i % len(cycle)] for i in range(N)]
    rgbs = list(map(_rgb, colors))

    # ---- Color mixing callback ----
    if isinstance(color_mixing, str):
        if color_mixing == "subtractive":
            mixing_cb = _color_mix_subtractive
        elif color_mixing == "average":
            mixing_cb = _color_mix_average
        elif color_mixing == "alpha":
            mixing_cb = _color_mix_alpha_stack
        else:
            raise ValueError(f"Unrecognized color_mixing string: {color_mixing!r}")
    elif callable(color_mixing):
        mixing_cb = color_mixing
    else:
        raise TypeError("color_mixing must be either a string or a callable.")

    # ---- Rasterize membership to image (RGBA) ----
    X, Y = geom["X"], geom["Y"]
    membership = [m.astype(float) for m in geom["membership"]]  # list (H,W)
    membership_stack = np.stack(membership, axis=-1)             # (H,W,N)

    # Disjoint region masks for both color mixing and label handling
    masks = _disjoint_region_masks([m.astype(bool) for m in membership])

    H, W, _ = membership_stack.shape
    rgba = np.zeros((H, W, 4), float)
    region_rgbs: Dict[Tuple[int, ...], np.ndarray] = {}

    for key, mask in masks.items():
        if not any(key):
            continue  # leave complement transparent
        if not mask.any():
            continue
        colors_for_key = [rgbs[i] for i, bit in enumerate(key) if bit]
        mixed_rgb = np.asarray(mixing_cb(colors_for_key), float)
        if mixed_rgb.shape != (3,):
            raise ValueError("color_mixing callback must return an RGB array of shape (3,).")
        region_rgbs[key] = mixed_rgb
        rgba[..., 0][mask] = mixed_rgb[0]
        rgba[..., 1][mask] = mixed_rgb[1]
        rgba[..., 2][mask] = mixed_rgb[2]
        rgba[..., 3][mask] = 1.0

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
    size_unit = float(geom["size_unit"])
    region_offsets = geom.get("region_offsets", {})
    region_label_method_default = geom.get("region_label_method_default", None)
    region_label_method_overrides = geom.get("region_label_method_overrides", {})

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
            eroded = binary_erosion(mask, structure=np.ones((3, 3), bool), border_value=0)
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
                # --- if all three sides within 1%, pick the one nearest 0°;
                #     else if two sides within 1%, select the third side; otherwise longest. ---
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
    # Convert text height (points) -> pixels -> data units (along vertical)
    def _data_units_for_pixels(ax, px: float) -> float:
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        p_disp = ax.transData.transform((cx, cy))
        p2_disp = (p_disp[0], p_disp[1] - px)
        p2_data = ax.transData.inverted().transform(p2_disp)
        return abs(p2_data[1] - cy)

    text_height_pts = float(region_label_fontsize)  # approximate
    text_height_px = text_height_pts * fig.dpi / 72.0
    down_len_data = 0.11 * _data_units_for_pixels(ax, text_height_px)

    # Auto text color helper (black/white) based on region RGB
    def _auto_text_color_from_rgb(rgb: np.ndarray) -> str:
        # Perceived luminance (Rec. 601)
        lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        return "white" if lum < 0.5 else "black"

    # Draw region texts
    for key, mask in masks.items():
        if not any(key):
            continue  # skip all-zeros region here (handled as complement below)
        if not mask.any():
            continue
        value = arr[key]
        if value is None:
            continue

        if N == 2:
            # N=2 special logic: region labels at the center of the line segment
            # that the shape cuts out from y=0.
            if Y.shape[0] > 1:
                dy_step = float(abs(Y[1, 0] - Y[0, 0]))
                eps_y = 0.75 * dy_step
            else:
                eps_y = 1e-9
            mask_line = mask & (np.abs(Y) <= eps_y)
            if mask_line.any():
                xs_line = X[mask_line]
                x_mid = 0.5 * (float(xs_line.min()) + float(xs_line.max()))
                pos = (x_mid, 0.0)
            else:
                # Fallback to centroid if region does not intersect y=0
                pos = _centroid(mask, X, Y)
                if pos is None:
                    continue
            dx = dy = 0.0
            rot_val = 0.0
            down_vec = np.array([0.0, 0.0], float)
        elif N == 1:
            pos = _centroid(mask, X, Y)
            if pos is None:
                continue
            dx = dy = 0.0
            rot_val = float(region_rotations.get(key, 0.0))
            theta = np.deg2rad(rot_val)
            down_vec = np.array([np.sin(theta), -np.cos(theta)], float) * down_len_data
        else:
            # N >= 3: choose between centroid and visual_center
            method = region_label_method_overrides.get(key, region_label_method_default or "centroid")
            if method == "visual_center":
                pos = _visual_center(mask, X, Y)
            else:
                pos = _centroid(mask, X, Y)
            if pos is None:
                continue
            dx, dy = region_offsets.get(tuple(map(int, key)), (0.0, 0.0))
            rot_val = float(region_rotations.get(key, 0.0))

            # Compute “down” direction in world coords for given rotation:
            theta = np.deg2rad(rot_val)
            down_vec = np.array([np.sin(theta), -np.cos(theta)], float) * down_len_data

        if text_color is None:
            rgb = region_rgbs.get(key)
            if rgb is not None:
                this_color = _auto_text_color_from_rgb(rgb)
            else:
                this_color = "black"
        else:
            this_color = text_color

        ax.text(
            pos[0] + dx + down_vec[0], pos[1] + dy + down_vec[1], f"{value}",
            ha="center", va="center", fontsize=region_label_fontsize, zorder=8,
            rotation=rot_val, rotation_mode="anchor", color=this_color
        )

    # ---- Complement (all-zeros) ----
    zeros = (0,) * N
    if arr[zeros] is not None:
        cx, cy = geom["complement_pos"]
        ax.text(cx, cy, f"{arr[zeros]}", ha="center", va="center", fontsize=region_label_fontsize)

    # ---- Class labels ----
    for (x, y, rot), name, col in zip(geom["label_positions"], class_names, rgbs):
        ax.text(x, y, name, ha="center", va="center", fontsize=class_name_fontsize,
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
        ax.set_title(title, fontsize=title_fontsize)

    if outfile:
        fig.savefig(outfile, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return None

    return fig
