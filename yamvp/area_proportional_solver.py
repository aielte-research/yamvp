import math
import time
import numbers
import numpy as np
from shapely.geometry import Polygon
from contextlib import contextmanager
from tqdm.auto import tqdm


# -------------------------------------------------------------
#  Simple aggregated timing framework
# -------------------------------------------------------------
TIMING_STATS = {}


def reset_timing_stats():
    """Reset global timing statistics."""
    global TIMING_STATS
    TIMING_STATS = {}


def record_timing(name, dt):
    """Accumulate time dt (seconds) for operation 'name'."""
    stat = TIMING_STATS.setdefault(name, {"count": 0, "total": 0.0})
    stat["count"] += 1
    stat["total"] += dt


@contextmanager
def timed(name):
    """Context manager to time a code block under a given operation name."""
    start = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - start
        record_timing(name, dt)


# -------------------------------------------------------------
#  Basic geometric utilities
# -------------------------------------------------------------
def circle_intersection_area(r1, r2, d):
    """Exact area of intersection of two circles with radii r1,r2 and center distance d."""
    r1 = float(r1)
    r2 = float(r2)
    d = float(d)

    # No overlap
    if d >= r1 + r2:
        return 0.0
    # One inside the other
    if d <= abs(r1 - r2):
        return math.pi * min(r1, r2) ** 2

    # Lens intersection
    alpha = 2 * math.acos((d**2 + r1**2 - r2**2) / (2 * r1 * d))
    beta = 2 * math.acos((d**2 + r2**2 - r1**2) / (2 * r2 * d))
    return (
        0.5 * r1**2 * (alpha - math.sin(alpha))
        + 0.5 * r2**2 * (beta - math.sin(beta))
    )


def solve_distance_for_overlap(r1, r2, target_area, tol=1e-6, max_iter=60):
    """
    Given radii r1, r2 and desired intersection area, solve for center distance d.
    Uses bisection with the exact 2-circle intersection formula.
    """
    r1 = float(r1)
    r2 = float(r2)
    target_area = float(target_area)

    if r1 <= 0 or r2 <= 0:
        return r1 + r2 + 1.0

    area_max = math.pi * min(r1, r2) ** 2

    if target_area <= 0:
        return r1 + r2 + 1e-6
    if target_area >= area_max:
        return abs(r1 - r2) + 1e-6

    d_min = abs(r1 - r2)
    d_max = r1 + r2

    for _ in range(max_iter):
        d_mid = 0.5 * (d_min + d_max)
        area_mid = circle_intersection_area(r1, r2, d_mid)
        if area_mid > target_area:
            d_min = d_mid
        else:
            d_max = d_mid
        if abs(area_mid - target_area) < tol:
            break

    return 0.5 * (d_min + d_max)


def solve_circles_2(values, base_radius=1.5):
    """
    Area-proportional 2-circle solver.

    values: 2x2 array-like (truth table order):
        [ [00, 01],
          [10, 11] ]

    - Treats None as 0.
    - Requires non-negative reals otherwise.
    - If at least two of {01,10,11} are 0/None, falls back to default geometry.
    - If AB (11) is 0/None but we do solve, circles are placed tangentially.
    - Returns (radA, radB, d, edge_case) or None if invalid / degenerate.
      edge_case is True when intersection is 0 or almost full containment,
      so the caller can skip fancy label geometry.
    """
    arr = np.asarray(values, dtype=object)
    if arr.shape != (2, 2):
        raise ValueError("solve_circles_2: values must have shape (2,2).")

    w = np.zeros_like(arr, dtype=float)
    types_ok = True

    # Copy values, treat None as 0, enforce numeric & non-negative
    for idx in np.ndindex(arr.shape):
        v = arr[idx]
        if v is None:
            w[idx] = 0.0
            continue
        if not isinstance(v, numbers.Real):
            types_ok = False
            break
        fv = float(v)
        if fv < 0.0:
            types_ok = False
            break
        w[idx] = fv

    if not types_ok:
        return None

    # Degenerate target: if at least two of the three non-complement regions
    # (01, 10, 11) are 0/None, bail out to original geometry.
    non_complement_indices = [(0, 1), (1, 0), (1, 1)]
    zero_count = sum(1 for idx in non_complement_indices if w[idx] <= 0.0)
    if zero_count > 1:
        return None

    w00, w01 = w[0, 0], w[0, 1]
    w10, w11 = w[1, 0], w[1, 1]

    # Set totals (A,B) and intersection
    SA = w10 + w11
    SB = w01 + w11
    S_inter = w11

    # If an entire class is zero, fall back to default geometry
    if SA <= 0.0 or SB <= 0.0:
        return None

    # Radii from totals (then scaled so max radius ≈ base_radius)
    rA_raw = math.sqrt(SA / math.pi)
    rB_raw = math.sqrt(SB / math.pi)
    r_max = max(rA_raw, rB_raw)
    if r_max <= 0.0:
        return None

    scale = float(base_radius) / r_max
    rA = rA_raw * scale
    rB = rB_raw * scale

    # Intersection area scale (areas ~ radius^2)
    S_inter_scaled = S_inter * (scale ** 2)

    # Distance and edge_case
    if S_inter_scaled <= 0.0:
        # AB is None or 0 → tangential circles
        d = rA + rB
        edge_case = True
    else:
        d = solve_distance_for_overlap(rA, rB, S_inter_scaled)
        edge_case = False
        # Flag "edge" cases (almost full containment) for label logic
        area_max = math.pi * min(rA, rB) ** 2
        if area_max > 0 and S_inter_scaled >= 0.95 * area_max:
            edge_case = True

    return rA, rB, d, edge_case


def sample_box_from_params(ellipse_params, margin_factor=0.2):
    """Bounding box for all ellipses with a margin."""
    ellipse_params = np.asarray(ellipse_params, dtype=float)
    cx = ellipse_params[:, 0]
    cy = ellipse_params[:, 1]
    a = np.abs(ellipse_params[:, 2])
    b = np.abs(ellipse_params[:, 3])
    r_bound = np.sqrt(a**2 + b**2)

    xmin = float(np.min(cx - r_bound))
    xmax = float(np.max(cx + r_bound))
    ymin = float(np.min(cy - r_bound))
    ymax = float(np.max(cy + r_bound))

    width = xmax - xmin
    height = ymax - ymin
    margin = margin_factor * max(width, height)

    return (xmin - margin, xmax + margin, ymin - margin, ymax + margin)


def generate_sample_points(box, n_samples=5000, rng=None):
    """Uniform random points in a box, plus box area."""
    if rng is None:
        rng = np.random.RandomState()

    xmin, xmax, ymin, ymax = box
    xs = rng.uniform(xmin, xmax, size=n_samples)
    ys = rng.uniform(ymin, ymax, size=n_samples)
    pts = np.stack([xs, ys], axis=1)
    box_area = (xmax - xmin) * (ymax - ymin)
    return pts, box_area


# -------------------------------------------------------------
#  Polygonal ellipse approximation + exact region areas
# -------------------------------------------------------------
def ellipse_to_polygon(cx, cy, a, b, theta, n_points=720):
    """Approximate an ellipse by a regular n-gon (Shapely polygon)."""
    with timed("ellipse_to_polygon"):
        t = np.linspace(0.0, 2.0 * math.pi, n_points, endpoint=False)
        cos_t = np.cos(t)
        sin_t = np.sin(t)

        cos_th = math.cos(theta)
        sin_th = math.sin(theta)

        x = cx + a * cos_t * cos_th - b * sin_t * sin_th
        y = cy + a * cos_t * sin_th + b * sin_t * cos_th
        coords = np.column_stack([x, y])
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
        return poly


def shapely_atomic_regions(ellipse_params, n_points=720):
    """
    Compute the 7 atomic regions for A,B,C using polygon Boolean ops.
    Returns (region_geoms, areas_dict) keyed by (1,0,0), ... ,(1,1,1).
    """
    with timed("shapely_atomic_regions"):
        ellipse_params = np.asarray(ellipse_params, dtype=float)
        cx = ellipse_params[:, 0]
        cy = ellipse_params[:, 1]
        a = np.abs(ellipse_params[:, 2])
        b = np.abs(ellipse_params[:, 3])
        theta = ellipse_params[:, 4]

        A_poly = ellipse_to_polygon(cx[0], cy[0], a[0], b[0], theta[0], n_points=n_points)
        B_poly = ellipse_to_polygon(cx[1], cy[1], a[1], b[1], theta[1], n_points=n_points)
        C_poly = ellipse_to_polygon(cx[2], cy[2], a[2], b[2], theta[2], n_points=n_points)

        B_union_C = B_poly.union(C_poly)
        A_union_C = A_poly.union(C_poly)
        A_union_B = A_poly.union(B_poly)

        AB = A_poly.intersection(B_poly)
        AC = A_poly.intersection(C_poly)
        BC = B_poly.intersection(C_poly)
        ABC = AB.intersection(C_poly)

        region_geoms = {}
        region_geoms[(1, 0, 0)] = A_poly.difference(B_union_C)
        region_geoms[(0, 1, 0)] = B_poly.difference(A_union_C)
        region_geoms[(0, 0, 1)] = C_poly.difference(A_union_B)
        region_geoms[(1, 1, 0)] = AB.difference(C_poly)
        region_geoms[(1, 0, 1)] = AC.difference(B_poly)
        region_geoms[(0, 1, 1)] = BC.difference(A_poly)
        region_geoms[(1, 1, 1)] = ABC

        areas_dict = {bits: geom.area for bits, geom in region_geoms.items()}
        return region_geoms, areas_dict


def components_of_geom(geom):
    """Number of connected components of a Shapely geometry (polygons only)."""
    if geom.is_empty:
        return 0
    gtype = geom.geom_type
    if gtype == "Polygon":
        return 1
    if gtype == "MultiPolygon":
        return len(geom.geoms)
    if gtype == "GeometryCollection":
        return sum(components_of_geom(g) for g in geom.geoms)
    return 1


# -------------------------------------------------------------
#  Weight / area vector helpers
# -------------------------------------------------------------
def weights_vector_from_tensor(w):
    """Vector W in the order [A, B, C, AB, AC, BC, ABC]."""
    w = np.asarray(w, dtype=float)
    return np.array(
        [
            w[1, 0, 0],
            w[0, 1, 0],
            w[0, 0, 1],
            w[1, 1, 0],
            w[1, 0, 1],
            w[0, 1, 1],
            w[1, 1, 1],
        ],
        dtype=float,
    )


def areas_vector_from_dict(areas_dict):
    """Vector A in the order [A, B, C, AB, AC, BC, ABC]."""
    return np.array(
        [
            areas_dict[(1, 0, 0)],
            areas_dict[(0, 1, 0)],
            areas_dict[(0, 0, 1)],
            areas_dict[(1, 1, 0)],
            areas_dict[(1, 0, 1)],
            areas_dict[(0, 1, 1)],
            areas_dict[(1, 1, 1)],
        ],
        dtype=float,
    )


# -------------------------------------------------------------
#  Monte Carlo region areas (still used for some init diagnostics)
# -------------------------------------------------------------
def region_areas_from_points(ellipse_params, points, box_area):
    with timed("region_areas_from_points"):
        ellipse_params = np.asarray(ellipse_params, dtype=float)
        cx = ellipse_params[:, 0]
        cy = ellipse_params[:, 1]
        a = np.abs(ellipse_params[:, 2])
        b = np.abs(ellipse_params[:, 3])
        theta = ellipse_params[:, 4]

        xs = points[:, 0]
        ys = points[:, 1]

        inside_list = []
        for i in range(3):
            dx = xs - cx[i]
            dy = ys - cy[i]
            cos_t = math.cos(theta[i])
            sin_t = math.sin(theta[i])
            x_prime = cos_t * dx + sin_t * dy
            y_prime = -sin_t * dx + cos_t * dy
            val = (x_prime / a[i]) ** 2 + (y_prime / b[i]) ** 2
            inside_list.append(val <= 1.0)
        inside = np.vstack(inside_list)

        codes = (
            (inside[0].astype(int) << 2)
            | (inside[1].astype(int) << 1)
            | inside[2].astype(int)
        )
        counts = np.bincount(codes, minlength=8).astype(float)
        areas = counts / len(points) * box_area

        areas_dict = {}
        for code in range(8):
            bits = ((code >> 2) & 1, (code >> 1) & 1, code & 1)
            areas_dict[bits] = areas[code]
        return areas_dict


# -------------------------------------------------------------
#  Initial 3-circle construction (Chow-style pivot variants)
# -------------------------------------------------------------
def initial_circles_from_weights(
    weights,
    n_phi_samples=20000,
    rng=None,
    run_label="",
    pivot="B",
):
    """
    Construction:
      - Pick radii from set totals.
      - Enforce correct pairwise intersection areas via circle distances.
      - Fix one circle (pivot) and revolve another around it to match triple area.
    """
    with timed("initial_circles_from_weights"):
        if rng is None:
            rng = np.random.RandomState()

        pivot = pivot.upper()
        if pivot not in ("A", "B", "C"):
            raise ValueError("pivot must be 'A', 'B', or 'C'")

        w = np.array(weights, dtype=float)

        names = {
            (1, 0, 0): "A only",
            (0, 1, 0): "B only",
            (0, 0, 1): "C only",
            (1, 1, 0): "A∩B only",
            (1, 0, 1): "A∩C only",
            (0, 1, 1): "B∩C only",
            (1, 1, 1): "A∩B∩C",
            (0, 0, 0): "outside",
        }

        region_w = {
            (1, 0, 0): w[1, 0, 0],
            (0, 1, 0): w[0, 1, 0],
            (0, 0, 1): w[0, 0, 1],
            (1, 1, 0): w[1, 1, 0],
            (1, 0, 1): w[1, 0, 1],
            (0, 1, 1): w[0, 1, 1],
            (1, 1, 1): w[1, 1, 1],
            (0, 0, 0): w[0, 0, 0],
        }

        # Unpack
        w100 = region_w[(1, 0, 0)]
        w010 = region_w[(0, 1, 0)]
        w001 = region_w[(0, 0, 1)]
        w110 = region_w[(1, 1, 0)]
        w101 = region_w[(1, 0, 1)]
        w011 = region_w[(0, 1, 1)]
        w111 = region_w[(1, 1, 1)]

        # Set totals
        SA = w100 + w110 + w101 + w111
        SB = w010 + w110 + w011 + w111
        SC = w001 + w101 + w011 + w111

        def radius_from_total(S):
            if S <= 0:
                return 0.3
            return math.sqrt(S / math.pi)

        rA = radius_from_total(SA)
        rB = radius_from_total(SB)
        rC = radius_from_total(SC)

        # Pairwise overlap targets
        TAB = w110 + w111
        TBC = w011 + w111
        TAC = w101 + w111

        dAB = solve_distance_for_overlap(rA, rB, TAB)
        dBC = solve_distance_for_overlap(rB, rC, TBC)
        dAC = solve_distance_for_overlap(rA, rC, TAC)

        def make_circle_params(centerA, centerB, centerC):
            """Helper to build [A,B,C] circle params from centers + radii."""
            return np.array(
                [
                    [centerA[0], centerA[1], rA, rA, 0.0],
                    [centerB[0], centerB[1], rB, rB, 0.0],
                    [centerC[0], centerC[1], rC, rC, 0.0],
                ],
                dtype=float,
            )

        # Place pivot and revolve the third
        if pivot == "B":
            centerB = np.array([0.0, 0.0])
            centerC = np.array([0.0, dBC])

            def make_params(centerA):
                return make_circle_params(centerA, centerB, centerC)

            revolve_radius = dAB
            pivot_center = centerB
        elif pivot == "A":
            centerA = np.array([0.0, 0.0])
            centerC = np.array([0.0, dAC])

            def make_params(centerB):
                return make_circle_params(centerA, centerB, centerC)

            revolve_radius = dAB
            pivot_center = centerA
        else:
            # pivot == "C"
            centerC = np.array([0.0, 0.0])
            centerB = np.array([0.0, dBC])

            def make_params(centerA):
                return make_circle_params(centerA, centerB, centerC)

            revolve_radius = dAC
            pivot_center = centerC

        target_triple = w111

        def triple_area(ellipse_params):
            box = sample_box_from_params(ellipse_params, margin_factor=0.2)
            pts, box_area = generate_sample_points(
                box, n_samples=n_phi_samples, rng=rng
            )
            areas_dict = region_areas_from_points(ellipse_params, pts, box_area)
            return areas_dict[(1, 1, 1)]

        best_phi = None
        best_err = None
        best_triple = None

        for phi in np.linspace(0.0, 2.0 * math.pi, 60, endpoint=False):
            center_revolved = pivot_center + np.array(
                [-revolve_radius * math.cos(phi), revolve_radius * math.sin(phi)]
            )
            ell = make_params(center_revolved)
            tri = triple_area(ell)
            err = abs(tri - target_triple)
            if best_err is None or err < best_err:
                best_err = err
                best_phi = phi
                best_triple = tri

        center_revolved_final = pivot_center + np.array(
            [-revolve_radius * math.cos(best_phi), revolve_radius * math.sin(best_phi)]
        )

        # --- FIXED BRANCH: build final circle params per pivot ----------------
        if pivot == "B":
            # A revolves around B; C fixed
            ellipse_params = make_circle_params(center_revolved_final, centerB, centerC)
        elif pivot == "A":
            # B revolves around A; C fixed
            ellipse_params = make_circle_params(centerA, center_revolved_final, centerC)
        else:  # pivot == "C"
            # A revolves around C; B fixed
            ellipse_params = make_circle_params(center_revolved_final, centerB, centerC)
        # ----------------------------------------------------------------------

        # MC estimate for debug (kept for potential external inspection)
        box = sample_box_from_params(ellipse_params, margin_factor=0.2)
        pts, box_area = generate_sample_points(
            box, n_samples=200000, rng=np.random.RandomState(42)
        )
        _ = region_areas_from_points(ellipse_params, pts, box_area)

        return ellipse_params


# -------------------------------------------------------------
#  Standard equilateral 3-circle initialization
# -------------------------------------------------------------
def standard_venn_initial_from_weights(weights):
    """Three equal circles at vertices of an equilateral triangle."""
    with timed("standard_venn_initial"):
        w = np.array(weights, dtype=float)

        w100 = w[1, 0, 0]
        w010 = w[0, 1, 0]
        w001 = w[0, 0, 1]
        w110 = w[1, 1, 0]
        w101 = w[1, 0, 1]
        w011 = w[0, 1, 1]
        w111 = w[1, 1, 1]

        SA = w100 + w110 + w101 + w111
        SB = w010 + w110 + w011 + w111
        SC = w001 + w101 + w011 + w111

        def radius_from_total(S):
            if S <= 0:
                return 0.3
            return math.sqrt(S / math.pi)

        rA = radius_from_total(SA)
        rB = radius_from_total(SB)
        rC = radius_from_total(SC)

        R_std = (rA + rB + rC) / 3.0
        d = R_std

        B = np.array([0.0, 0.0])
        C = np.array([d, 0.0])
        A = np.array([0.5 * d, (math.sqrt(3) / 2.0) * d])

        ellipse_params = np.array(
            [
                [A[0], A[1], R_std, R_std, 0.0],
                [B[0], B[1], R_std, R_std, 0.0],
                [C[0], C[1], R_std, R_std, 0.0],
            ],
            dtype=float,
        )
        return ellipse_params


# -------------------------------------------------------------
#  Long-ellipse equilateral initialization for ABC == 0
# -------------------------------------------------------------
def long_ellipse_equilateral_initial(aspect_ratio=3.0):
    """
    Three long ellipses with centers at the vertices of an equilateral triangle.
    All have aspect ratio a/b = aspect_ratio (>= 5), and are oriented so their
    major axes point towards the triangle centroid. This configuration tends to
    strongly de-emphasize the triple-overlap region from the start.
    """
    with timed("long_ellipse_equilateral_initial"):
        # Base minor radius and aspect ratio
        b = 1.0
        a = float(aspect_ratio) * b

        # Side length chosen so that distance from each vertex to centroid
        # equals the major radius a, i.e. L / sqrt(3) = a -> L = a * sqrt(3)
        L = a * math.sqrt(3.0)/2

        B = np.array([0.0, 0.0])
        C = np.array([L, 0.0])
        A = np.array([0.5 * L, (math.sqrt(3.0) / 2.0) * L])

        centers = [A, B, C]
        centers_arr = np.vstack(centers)
        grand_center = centers_arr.mean(axis=0)

        ellipse_params = np.zeros((3, 5), dtype=float)
        for idx, center in enumerate(centers):
            vec = grand_center - center
            angle = math.atan2(vec[1], vec[0])  # radians, major axis towards centroid
            ellipse_params[idx] = [center[0], center[1], a, b, angle+math.pi/2.0]

        return ellipse_params


# -------------------------------------------------------------
#  eulerAPE-style starting state (approx)
# -------------------------------------------------------------
def eulerape_style_initial_from_weights(
    weights,
    rng=None,
    n_line_samples=81,
):
    """
    eulerAPE-style starting diagram:
      - choose the two largest sets and draw a 2-circle area-proportional
        diagram for them using a Chow-Ruskey-style distance;
      - place the third circle along the angle-bisector line from one
        intersection point, searching along that line to match triple area.
    """
    with timed("eulerAPE-style_initial"):
        if rng is None:
            rng = np.random.RandomState()

        w = np.array(weights, dtype=float)

        # Region weights
        w100 = w[1, 0, 0]
        w010 = w[0, 1, 0]
        w001 = w[0, 0, 1]
        w110 = w[1, 1, 0]
        w101 = w[1, 0, 1]
        w011 = w[0, 1, 1]
        w111 = w[1, 1, 1]

        SA = w100 + w110 + w101 + w111
        SB = w010 + w110 + w011 + w111
        SC = w001 + w101 + w011 + w111

        def radius_from_total(S):
            if S <= 0:
                return 0.3
            return math.sqrt(S / math.pi)

        rA = radius_from_total(SA)
        rB = radius_from_total(SB)
        rC = radius_from_total(SC)

        radii = np.array([rA, rB, rC], dtype=float)
        S_vec = np.array([SA, SB, SC], dtype=float)

        # Order sets by decreasing total: e1,e2,e3
        order = np.argsort(S_vec)[::-1]
        e1, e2, e3 = order[0], order[1], order[2]

        # Pairwise overlap target for e1,e2
        if (e1, e2) in ((0, 1), (1, 0)):
            T12 = w110 + w111
        elif (e1, e2) in ((0, 2), (2, 0)):
            T12 = w101 + w111
        else:
            T12 = w011 + w111

        r1 = radii[e1]
        r2 = radii[e2]
        r3 = radii[e3]

        d12 = solve_distance_for_overlap(r1, r2, T12)

        # If circles do not intersect in two points, fall back
        if not (abs(r1 - r2) < d12 < r1 + r2):
            return initial_circles_from_weights(
                weights,
                rng=rng,
                n_phi_samples=4000,
                run_label="eulerAPE-style fallback",
                pivot="B",
            )

        # Place e1 at (0,0), e2 at (d12,0)
        c1 = np.array([0.0, 0.0])
        c2 = np.array([d12, 0.0])

        # Intersection point (upper) of the two circles
        d = d12
        x0 = (r1**2 - r2**2 + d**2) / (2 * d)
        y_sq = r1**2 - x0**2
        if y_sq < 0:
            y_sq = 0.0
        y0 = math.sqrt(y_sq)
        P = np.array([x0, y0])

        # Tangent directions at P (perpendicular to radius vectors)
        v1 = P - c1
        v2 = P - c2
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            return initial_circles_from_weights(
                weights,
                rng=rng,
                n_phi_samples=4000,
                run_label="eulerAPE-style fallback",
                pivot="B",
            )

        t1 = np.array([-v1[1], v1[0]], dtype=float)
        t2 = np.array([-v2[1], v2[0]], dtype=float)
        t1 = t1 / np.linalg.norm(t1)
        t2 = t2 / np.linalg.norm(t2)

        v_bis = t1 + t2
        norm_v = np.linalg.norm(v_bis)
        if norm_v == 0:
            v_bis = t1
        else:
            v_bis = v_bis / norm_v

        # Search along line P + t * v_bis
        t_range = 3.0 * max(r1, r2, r3)
        t_values = np.linspace(-t_range, t_range, n_line_samples)

        best_t = None
        best_err = None
        best_triple = None
        target_triple = w111

        def triple_area_exact(c3):
            ellipse_params = np.array(
                [
                    [c1[0], c1[1], r1, r1, 0.0],
                    [c2[0], c2[1], r2, r2, 0.0],
                    [c3[0], c3[1], r3, r3, 0.0],
                ],
                dtype=float,
            )
            _, areas_dict = shapely_atomic_regions(ellipse_params, n_points=360)
            return areas_dict[(1, 1, 1)]

        for t_val in t_values:
            c3 = P + t_val * v_bis
            d13 = np.linalg.norm(c3 - c1)
            d23 = np.linalg.norm(c3 - c2)
            # require two-point intersections with both circles
            if not (abs(r1 - r3) < d13 < r1 + r3):
                continue
            if not (abs(r2 - r3) < d23 < r2 + r3):
                continue
            tri = triple_area_exact(c3)
            err = abs(tri - target_triple)
            if best_err is None or err < best_err:
                best_err = err
                best_t = t_val
                best_triple = tri

        if best_t is None:
            return initial_circles_from_weights(
                weights,
                rng=rng,
                n_phi_samples=4000,
                run_label="eulerAPE-style fallback",
                pivot="B",
            )

        c3 = P + best_t * v_bis

        # Map centers back to A,B,C order
        centers = [None, None, None]
        centers[e1] = c1
        centers[e2] = c2
        centers[e3] = c3

        # Fixed rotations like in eulerAPE (0, π/3, 2π/3)
        angles = [0.0, math.pi / 3.0, 2.0 * math.pi / 3.0]

        ellipse_params = np.zeros((3, 5), dtype=float)
        radii_list = [rA, rB, rC]
        for idx in range(3):
            cx, cy = centers[idx]
            r = radii_list[idx]
            ellipse_params[idx] = [cx, cy, r, r, angles[idx]]

        return ellipse_params


# -------------------------------------------------------------
#  Loss function (ratio) + disjointness penalty
# -------------------------------------------------------------
def venn_loss(
    params_flat,
    weights,
    loss_type="ratio",
    conn_weight=0.1,
    poly_points=720,
    return_components=False,
):
    """
    Total loss: ratio-based mismatch + conn_weight * disjointness penalty.

    Revised core per-region term t[i]:

      - If target ~0 and area ~0: t[i] = 0
      - If target ~0 but area > 0: t[i] = 1 + A_norm[i]
      - If target > 0 but area ~0: t[i] = 1
      - If both > 0:               t[i] = log(A_norm[i] / W_norm[i])

    This makes the normalized area itself part of the loss and handles
    zero / near-zero targets robustly without ad-hoc penalties.
    """
    with timed("venn_loss"):
        ellipse_params = np.asarray(params_flat, dtype=float).reshape(3, 5)

        W = weights_vector_from_tensor(weights)
        W_sum = W.sum()
        if W_sum <= 0:
            total_loss = 0.0
            if not return_components:
                return total_loss
            W_norm = np.zeros_like(W)
            A = np.zeros_like(W)
            A_norm = np.zeros_like(W)
            t = np.zeros_like(W)
            ratio = np.zeros_like(W)
            areas_dict = {
                (1, 0, 0): 0.0,
                (0, 1, 0): 0.0,
                (0, 0, 1): 0.0,
                (1, 1, 0): 0.0,
                (1, 0, 1): 0.0,
                (0, 1, 1): 0.0,
                (1, 1, 1): 0.0,
            }
            components = {
                "targets": W_norm,
                "areas": A,
                "areas_norm": A_norm,
                "ratio": ratio,
                "t": t,
                "areas_dict": areas_dict,
            }
            return total_loss, components

        W_norm = W / W_sum

        region_geoms, areas_dict = shapely_atomic_regions(
            ellipse_params, n_points=poly_points
        )
        A = areas_vector_from_dict(areas_dict)
        A_sum = A.sum()
        if A_sum <= 0:
            total_loss = 1e6
            if not return_components:
                return total_loss
            A_norm = np.zeros_like(A)
            t = np.zeros_like(A_norm)
            ratio = np.zeros_like(A_norm)
            components = {
                "targets": W_norm,
                "areas": A,
                "areas_norm": A_norm,
                "ratio": ratio,
                "t": t,
                "areas_dict": areas_dict,
            }
            return total_loss, components

        A_norm = A / A_sum

        # Base loss: robust log-ratio / area-based
        if loss_type != "ratio":
            raise ValueError("Only 'ratio' loss_type is supported now.")

        eps = 1e-8
        t = np.zeros_like(W_norm)

        for i in range(len(W_norm)):
            w = W_norm[i]
            a = A_norm[i]

            if w <= eps and a <= eps:
                # target ~0, area ~0 → perfect match
                t[i] = 0.0
            elif w <= eps and a > eps:
                # target ~0 (or exactly 0), but region exists:
                # penalize directly by normalized area
                t[i] = 1.0 + a
            elif w > eps and a <= eps:
                # target non-zero but region vanished:
                # pure "100% off" type error
                t[i] = 1.0
            else:
                # both positive: ratio mismatch (log-symmetric for over/under)
                t[i] = np.log(a / w)

        base_loss = float(np.sqrt(np.sum(t**2)))

        # Disjointness penalty (unchanged)
        disjoint_penalty = 0.0

        for geom in region_geoms.values():
            if geom.is_empty:
                continue

            polys = None
            gtype = geom.geom_type
            if gtype == "Polygon":
                polys = [geom]
            elif gtype == "MultiPolygon":
                polys = list(geom.geoms)
            elif gtype == "GeometryCollection":
                polys = [g for g in geom.geoms if isinstance(g, Polygon)]
            else:
                polys = []

            if polys is None or len(polys) <= 1:
                continue

            areas = np.array([p.area for p in polys if p.area > 0.0], dtype=float)
            if areas.size <= 1:
                continue

            total_area = float(areas.sum())
            if total_area <= 0.0:
                continue

            norm_areas = areas / total_area
            prod = float(np.prod(norm_areas))
            disjoint_penalty += prod

        total_loss = base_loss + conn_weight * disjoint_penalty

        if not return_components:
            return total_loss

        ratio = np.zeros_like(A_norm)
        np.divide(
            A_norm,
            W_norm,
            out=ratio,
            where=W_norm > eps,
        )

        components = {
            "targets": W_norm,
            "areas": A,
            "areas_norm": A_norm,
            "ratio": ratio,
            "t": t,
            "areas_dict": areas_dict,
        }
        return total_loss, components


# -------------------------------------------------------------
#  Evolutionary random search optimizer (with LR & poly_points schedule)
# -------------------------------------------------------------
def optimize_ellipses(
    weights,
    init_params,
    n_iters=30,
    lr_start=0.1,
    lr_end=0.01,
    loss_type="ratio",
    conn_weight=0.1,
    poly_points=720,
    poly_points_start=None,
    n_candidates=20,
    rng=None,
    verbose=True,
):
    with timed("optimize_ellipses"):
        if rng is None:
            rng = np.random.RandomState()

        params_flat = np.asarray(init_params, dtype=float).reshape(-1)

        if poly_points_start is None:
            poly_points_start = poly_points

        losses = []
        params_history = [init_params.copy()]

        if verbose:
            iterator = tqdm(
                range(n_iters),
                desc=f"Optimizing ({loss_type})",
                unit="iter",
                leave=False,
            )
        else:
            iterator = range(n_iters)

        for it in iterator:
            if n_iters > 1:
                progress = it / (n_iters - 1)
            else:
                progress = 1.0
            lr_t = math.exp(math.log(lr_start) * (1 - progress) + math.log(lr_end) * progress)

            # Gradually refine polygon resolution from coarse to fine
            poly_points_t = int(
                round(poly_points_start + (poly_points - poly_points_start) * progress)
            )
            poly_points_t = max(4, poly_points_t)

            def loss_fn(p_flat):
                return venn_loss(
                    p_flat,
                    weights,
                    loss_type=loss_type,
                    conn_weight=conn_weight,
                    poly_points=poly_points_t,
                )

            current_loss = loss_fn(params_flat)
            best_loss = current_loss
            best_params = params_flat.copy()

            scales = np.maximum(np.abs(params_flat), 1.0)

            for _ in range(n_candidates):
                noise = rng.normal(size=params_flat.shape)
                proposal = params_flat + lr_t * scales * noise
                prop_mat = proposal.reshape(3, 5)
                for i in range(3):
                    prop_mat[i, 2] = max(prop_mat[i, 2], 0.3)
                    prop_mat[i, 3] = max(prop_mat[i, 3], 0.3)
                proposal = prop_mat.reshape(-1)

                loss_p = loss_fn(proposal)
                if loss_p < best_loss:
                    best_loss = loss_p
                    best_params = proposal

            params_flat = best_params
            losses.append(best_loss)
            params_history.append(params_flat.reshape(3, 5))

            if verbose:
                iterator.set_postfix(loss=best_loss, lr=lr_t, poly=poly_points_t)

        return params_flat.reshape(3, 5), np.array(losses), params_history


# -------------------------------------------------------------
#  Quality metric via per-ellipse thinning/thickening
# -------------------------------------------------------------
def any_unconnected_for_params(ellipse_params, poly_points=360, area_eps=1e-6):
    with timed("any_unconnected_for_params"):
        region_geoms, _ = shapely_atomic_regions(ellipse_params, n_points=poly_points)
        for geom in region_geoms.values():
            if geom.is_empty:
                continue
            if geom.area < area_eps:
                continue
            comps = components_of_geom(geom)
            if comps > 1:
                return True
        return False


def scale_one_ellipse(ellipse_params, idx, s):
    ep = np.asarray(ellipse_params, dtype=float).copy()
    ep[idx, 2] *= s
    ep[idx, 3] *= s
    return ep


def compute_quality(
    ellipse_params,
    poly_points=360,
    area_eps=1e-6,
    s_min=0.1,
    s_max=3.0,
    n_samples=8,
    n_bisect=8,
):
    """
    Quality Q(Θ):
      - If initial config is connected, Q>0 is how much you can shrink each ellipse
        (min over ellipses) before any region becomes disconnected.
      - If already unconnected, Q<0 is minus how much you have to grow at least one
        ellipse to make all regions connected.
    """
    with timed("compute_quality"):
        base = np.asarray(ellipse_params, dtype=float).reshape(3, 5)
        base_unconnected = any_unconnected_for_params(
            base, poly_points=poly_points, area_eps=area_eps
        )

        qualities = []
        for idx in range(3):
            if not base_unconnected:
                # Shrink
                scales = np.linspace(1.0, s_min, n_samples)
                last_s = scales[0]
                last_state = False
                bracket_found = False
                lo = None
                hi = None
                for s in scales[1:]:
                    ep_s = scale_one_ellipse(base, idx, s)
                    state = any_unconnected_for_params(
                        ep_s, poly_points=poly_points, area_eps=area_eps
                    )
                    if (not last_state) and state:
                        hi = last_s
                        lo = s
                        bracket_found = True
                        break
                    last_s = s
                    last_state = state
                if not bracket_found:
                    q_i = 1.0
                else:
                    for _ in range(n_bisect):
                        mid = 0.5 * (lo + hi)
                        ep_mid = scale_one_ellipse(base, idx, mid)
                        state = any_unconnected_for_params(
                            ep_mid, poly_points=poly_points, area_eps=area_eps
                        )
                        if state:
                            lo = mid
                        else:
                            hi = mid
                    s_break = hi
                    q_i = 1.0 - s_break
            else:
                # Grow
                scales = np.linspace(1.0, s_max, n_samples)
                last_s = scales[0]
                last_state = True
                bracket_found = False
                lo = None
                hi = None
                for s in scales[1:]:
                    ep_s = scale_one_ellipse(base, idx, s)
                    state = any_unconnected_for_params(
                        ep_s, poly_points=poly_points, area_eps=area_eps
                    )
                    if last_state and (not state):
                        lo = last_s
                        hi = s
                        bracket_found = True
                        break
                    last_s = s
                    last_state = state
                if not bracket_found:
                    q_i = -(s_max - 1.0)
                else:
                    for _ in range(n_bisect):
                        mid = 0.5 * (lo + hi)
                        ep_mid = scale_one_ellipse(base, idx, mid)
                        state = any_unconnected_for_params(
                            ep_mid, poly_points=poly_points, area_eps=area_eps
                        )
                        if state:
                            lo = mid
                        else:
                            hi = mid
                    s_fix = hi
                    growth = s_fix - 1.0
                    q_i = -growth
            qualities.append(q_i)

        return min(qualities)


# -------------------------------------------------------------
#  High-level solver for 3 ellipses
# -------------------------------------------------------------
def solve_ellipses_3(
    weights,
    loss_type="ratio",
    lr_start=0.01,
    lr_end=0.0001,
    n_iters=1000,
    conn_weight=1.0,
    poly_points=1000,
    poly_points_start=100,
    init_phi_samples=4000,
    n_candidates=10,
    quality_poly_points=2000,
    quality_n_samples=20,
    loss_threshold=0.01,
    rng=None,
    use_standard=True,
    use_pivotA=True,
    use_pivotB=True,
    use_pivotC=True,
    use_eulerape=True,
    verbose=True,
):
    """
    Run multiple initializations, optimize each, and select the best solution.

    Steps:
      1) Generate several initial ellipse configs.
      2) Optimize each with a coarse->fine poly_points schedule.
      3) Recompute final loss with high-resolution polygons (poly_points).
      4) Filter to candidates with loss < loss_threshold.
      5) For those, compute a precise quality.
      6) Pick the config with highest quality (tie-breaker: lowest loss).
      7) Build a detailed history for the chosen run:
         - params, loss, areas, and per-region loss components.

    Returns:
      None  # if no config has loss < loss_threshold
      or
      (best_params, best_loss, best_quality, history)

      - best_params: np.ndarray, shape (3, 5)
      - best_loss: float (high-res loss)
      - best_quality: float (high-res quality)
      - history: list of dicts, one per iteration, with keys:
          * "iter": iteration index
          * "params": (3, 5) ellipse parameters
          * "loss": float
          * "areas": dict from (bits) -> area
          * "loss_components": dict with:
                - "targets": W_norm
                - "areas_norm": A_norm
                - "ratio": A_norm / W_norm
                - "t": per-region loss terms
    """
    if rng is None:
        rng = np.random.RandomState()

    w_arr = np.asarray(weights, dtype=float)
    w111 = float(w_arr[1, 1, 1])
    triple_zero = abs(w111) < 1e-12

    init_params_list = []
    labels = []

    if use_standard:
        init_params_list.append(standard_venn_initial_from_weights(weights))
        labels.append("Standard Venn")

    if use_pivotA:
        init_params_list.append(
            initial_circles_from_weights(
                weights,
                rng=rng,
                n_phi_samples=init_phi_samples,
                run_label="solve_ellipses_3, pivot A",
                pivot="A",
            )
        )
        labels.append("Algo pivot A")

    if use_pivotB:
        init_params_list.append(
            initial_circles_from_weights(
                weights,
                rng=rng,
                n_phi_samples=init_phi_samples,
                run_label="solve_ellipses_3, pivot B",
                pivot="B",
            )
        )
        labels.append("Algo pivot B")

    if use_pivotC:
        init_params_list.append(
            initial_circles_from_weights(
                weights,
                rng=rng,
                n_phi_samples=init_phi_samples,
                run_label="solve_ellipses_3, pivot C",
                pivot="C",
            )
        )
        labels.append("Algo pivot C")

    if use_eulerape:
        init_params_list.append(
            eulerape_style_initial_from_weights(
                weights,
                rng=rng,
                n_line_samples=81,
            )
        )
        labels.append("eulerAPE-style")

    # Extra long-ellipse equilateral start, only when ABC target is 0
    if triple_zero:
        init_params_list.append(long_ellipse_equilateral_initial(aspect_ratio=5.0))
        labels.append("Long equilateral (ABC=0)")

    opt_params_list = []
    params_hist_list = []
    final_losses_precise = []

    n_inits = len(init_params_list)
    if verbose:
        init_iterator = tqdm(
            range(n_inits),
            desc="solve_ellipses_3 inits",
            unit="init",
        )
    else:
        init_iterator = range(n_inits)

    for idx in init_iterator:
        init_params = init_params_list[idx]
        opt_params, _, params_hist = optimize_ellipses(
            weights,
            init_params,
            n_iters=n_iters,
            lr_start=lr_start,
            lr_end=lr_end,
            loss_type=loss_type,
            conn_weight=conn_weight,
            poly_points=poly_points,
            poly_points_start=poly_points_start,
            n_candidates=n_candidates,
            rng=rng,
            verbose=verbose,  # suppress inner tqdm; outer one is enough
        )
        opt_params_list.append(opt_params)
        params_hist_list.append(params_hist)

        # High-res loss for final candidate
        final_loss, _ = venn_loss(
            opt_params.reshape(-1),
            weights,
            loss_type=loss_type,
            conn_weight=conn_weight,
            poly_points=poly_points,
            return_components=True,
        )
        final_losses_precise.append(final_loss)

    # Filter candidates by loss threshold
    final_losses_precise = np.array(final_losses_precise, dtype=float)
    candidate_indices = [
        i for i, L in enumerate(final_losses_precise) if L < loss_threshold
    ]

    if len(candidate_indices) == 0:
        return None

    # Compute quality only for those candidates under threshold (more precise)
    qualities = []
    for i in candidate_indices:
        q = compute_quality(
            opt_params_list[i],
            poly_points=quality_poly_points,
            n_samples=quality_n_samples,
        )
        qualities.append(q)

    # Pick best: highest quality, break ties by lower loss
    best_idx = None
    best_quality = None
    best_loss = None

    for j, i in enumerate(candidate_indices):
        q = qualities[j]
        L = final_losses_precise[i]
        if best_idx is None or q > best_quality or (
            best_quality is not None and math.isclose(q, best_quality) and L < best_loss
        ):
            best_idx = i
            best_quality = q
            best_loss = L

    best_params = opt_params_list[best_idx]
    best_params_hist = params_hist_list[best_idx]

    # Build detailed history for chosen run using high-res loss/components
    history = []
    for it, params in enumerate(best_params_hist):
        loss_it, comps = venn_loss(
            params.reshape(-1),
            weights,
            loss_type=loss_type,
            conn_weight=conn_weight,
            poly_points=poly_points,
            return_components=True,
        )
        history.append(
            {
                "iter": it,
                "params": params,
                "loss": loss_it,
                "areas": comps["areas_dict"],
                "loss_components": {
                    "targets": comps["targets"],
                    "areas_norm": comps["areas_norm"],
                    "ratio": comps["ratio"],
                    "t": comps["t"],
                },
            }
        )

    return best_params, float(best_loss), float(best_quality), history
