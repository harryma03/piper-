"""Unified grasp candidate generation, re-validation and ranking.

Every candidate -- network or geometric, original or rotated -- passes the
same validation against the observed point cloud before execution:

1. width feasibility: the local object width along the opening axis must fit
   the simulated Piper opening (70 mm);
2. jaw containment: the target must occupy the predicted jaw volume;
3. bilateral support: both fingers must see plausible contact material.

Network scores are a ranking prior only, never a success probability. The
combined ``confidence`` is a documented heuristic for ordering, not a
calibrated probability. Unobserved surfaces stay unobserved: nothing here
completes or trusts a full object model.

Candidate dicts carry at least: frame, stamp, T_base_tcp / T_base_tip /
T_base_pregrasp_tcp, opening_m, predicted_width_m, approach_direction,
approach_distance_m, source, method, score, confidence, validation,
rejection_reason.
"""
import math

import numpy as np

from piperx_lab.grasp_candidate_filter import GRASPNET_FINGER_LENGTH_M

# Simulated Piper gripper envelope. These are target widths, not forces.
MAX_OPENING_M = 0.070
OPENING_MARGIN_M = 0.004
TIP_FRAME_TO_WORLD = 'base_link'
# Tip frame axes after the GraspNet->Piper alignment: +Z approach,
# +Y opening, +X transverse (see graspnet_adapter.convert_grasps).
APPROACH_AXIS = 2
OPENING_AXIS = 1
TRANSVERSE_AXIS = 0
GRIP_REGION_LENGTH_M = 0.02      # |z| window around the tip for contact checks
WIDTH_ROBUST_Q = (0.05, 0.95)    # robust width quantiles
MIN_SIDE_POINTS = 3              # per-finger material for bilateral support


def tip_frame_axes(T_base_tip):
    """Return (approach, opening, transverse) unit vectors in base frame."""
    rotation = np.asarray(T_base_tip, dtype=float)[:3, :3]
    return rotation[:, APPROACH_AXIS], rotation[:, OPENING_AXIS], rotation[:, TRANSVERSE_AXIS]


def local_width_along_opening(points, T_base_tip, window_length_m=GRIP_REGION_LENGTH_M,
                              transverse_half_m=0.03, quantiles=WIDTH_ROBUST_Q):
    """Robust object width across the opening axis near the tip.

    Points within the grip window (|z| and |x| bounded in tip frame) are
    projected on the opening axis; the width is the p95-p5 spread, which
    tolerates stray points better than ptp.
    """
    points = np.asarray(points, dtype=float)
    tip_position = np.asarray(T_base_tip, dtype=float)[:3, 3]
    approach, opening, transverse = tip_frame_axes(T_base_tip)
    offsets = points - tip_position
    z = offsets @ approach
    x = offsets @ transverse
    y = offsets @ opening
    window = (np.abs(z) <= window_length_m) & (np.abs(x) <= transverse_half_m)
    if int(window.sum()) < 6:
        return None, 0
    width = float(np.ptp(np.quantile(y[window], quantiles)))
    return width, int(window.sum())


def revalidate_candidate(candidate, points_base, *, max_opening_m=MAX_OPENING_M,
                         opening_margin_m=OPENING_MARGIN_M,
                         min_jaw_points=6, jaw_length_m=GRASPNET_FINGER_LENGTH_M,
                         min_bilateral_points=MIN_SIDE_POINTS, hull_vertices=None,
                         table_height_m=None):
    """Re-check one candidate against the observed base-frame cloud.

    Returns ``{'ok': bool, 'reason': str|None, 'validation': {...}}`` and
    never raises on geometric mismatch: rejection reasons are data.

    The local width is taken from the cloud AND, when available, from the
    completed hull vertices: a partially visible arc under-reports the width
    of round objects, and an under-reported width would command an
    over-squeezing close. The larger of the two estimates wins, which is
    conservative for both feasibility and closing.

    ``table_height_m`` enables a reachability pre-check: the pregrasp flange
    trails the tip by the 142.5 mm model-tip offset along the approach, so
    upward-scooping approaches put the wrist below the support plane where
    IK can never succeed; such candidates are rejected here instead of
    burning planning attempts.
    """
    validation = {'checked': True}
    tip = np.asarray(candidate.get('T_base_tip'), dtype=float)
    if tip.shape != (4, 4) or not np.all(np.isfinite(tip)):
        return {'ok': False, 'reason': 'missing_T_base_tip', 'validation': validation}
    pregrasp = np.asarray(candidate.get('T_base_pregrasp_tcp'), dtype=float)
    if pregrasp.shape != (4, 4) or not np.all(np.isfinite(pregrasp)):
        return {'ok': False, 'reason': 'missing_T_base_pregrasp_tcp', 'validation': validation}
    points = np.asarray(points_base, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 20:
        return {'ok': False, 'reason': 'insufficient_observed_points', 'validation': validation}
    opening = float(candidate.get('opening_m', float('nan')))
    if not math.isfinite(opening) or not 0 < opening <= max_opening_m:
        return {'ok': False, 'reason': 'opening_out_of_range', 'validation': validation}
    if table_height_m is not None and pregrasp[2, 3] < table_height_m + 0.005:
        return {'ok': False, 'reason': 'pregrasp_wrist_below_support_plane',
                'validation': validation}

    approach, opening_axis, transverse = tip_frame_axes(tip)
    tip_position = tip[:3, 3]
    offsets = points - tip_position
    z = offsets @ approach
    y = offsets @ opening_axis
    x = offsets @ transverse

    # 1. Local width along the opening axis near the grip point.
    near = (np.abs(z) <= GRIP_REGION_LENGTH_M) & (np.abs(x) <= 0.03)
    if int(near.sum()) < 6:
        return {'ok': False, 'reason': 'no_target_material_at_grip_point',
                'validation': validation}
    local_width = float(np.ptp(np.quantile(y[near], WIDTH_ROBUST_Q)))
    hull_width = 0.0
    if hull_vertices is not None and len(hull_vertices) >= 4:
        hull_vertices = np.asarray(hull_vertices, dtype=float)
        hull_offsets = hull_vertices - tip_position
        hull_window = ((np.abs(hull_offsets @ approach) <= 0.03) &
                       (np.abs(hull_offsets @ transverse) <= 0.04))
        if int(hull_window.sum()) >= 3:
            hull_width = float(np.ptp(np.quantile(
                hull_offsets[hull_window] @ opening_axis, WIDTH_ROBUST_Q)))
    local_width = max(local_width, hull_width)
    validation['predicted_width_m'] = local_width
    validation['hull_width_m'] = hull_width
    required_opening = local_width + opening_margin_m
    if required_opening > max_opening_m:
        return {'ok': False, 'reason': 'width_infeasible_for_piper_opening',
                'validation': validation}

    # 2. Jaw containment over the full finger length behind the tip.
    jaw = ((z >= -jaw_length_m - 0.005) & (z <= GRIP_REGION_LENGTH_M) &
           (np.abs(y) <= opening / 2 + 0.005) & (np.abs(x) <= 0.045))
    jaw_points = int(jaw.sum())
    validation['jaw_points'] = jaw_points
    if jaw_points < min_jaw_points:
        return {'ok': False, 'reason': 'insufficient_target_in_jaw_volume',
                'validation': validation}
    containment = float((near & (np.abs(y) <= local_width / 2 + 0.008)).sum()) / max(int(near.sum()), 1)
    validation['containment_fraction'] = containment

    # 3. Bilateral support: material on both sides of the opening axis plane.
    contact_band = near & (np.abs(x) <= 0.03) & (np.abs(z) <= GRIP_REGION_LENGTH_M)
    left = int((contact_band & (y <= -0.002)).sum())
    right = int((contact_band & (y >= 0.002)).sum())
    validation['side_points'] = [left, right]
    if min(left, right) < min_bilateral_points:
        return {'ok': False, 'reason': 'insufficient_bilateral_contact_support',
                'validation': validation}
    balance = min(left, right) / max(max(left, right), 1)
    validation['bilateral_balance'] = float(balance)

    # Heuristic ordering score, not a probability.
    width_margin = 1.0 - required_opening / max_opening_m
    score = float(candidate.get('score', 0.0) or 0.0)
    score_norm = math.tanh(max(score, 0.0))
    confidence = (0.35 * score_norm + 0.25 * containment +
                  0.2 * balance + 0.2 * max(0.0, width_margin))
    validation['confidence'] = float(confidence)
    return {'ok': True, 'reason': None, 'validation': validation}


def apply_revalidation(candidates, points_base, hull_vertices=None, **kwargs):
    """Validate a candidate list; keeps every candidate annotated.

    Valid candidates gain ``predicted_width_m``/``confidence``/``validation``
    and ``rejection_reason=None``; invalid ones stay in the list with
    ``rejection_reason`` set so batch reports can count rejection causes.
    Callers execute only the ``ok`` ones, ordered by confidence.
    """
    annotated = []
    for candidate in candidates:
        item = dict(candidate)
        result = revalidate_candidate(item, points_base, hull_vertices=hull_vertices, **kwargs)
        validation = dict(item.get('validation') or {})
        validation.update(result['validation'])
        item['validation'] = validation
        item['rejection_reason'] = result['reason']
        if result['ok']:
            item['predicted_width_m'] = result['validation'].get('predicted_width_m')
            item['confidence'] = result['validation'].get('confidence')
        annotated.append(item)
    return annotated


def _approach_rotation(approach, opening):
    """Right-handed tip rotation with +Z approach and +Y opening.

    The transverse axis must be ``Y x Z`` (not ``Z x Y``, which yields a
    left-handed frame with determinant -1).
    """
    approach = np.asarray(approach, float)
    approach /= np.linalg.norm(approach)
    opening = np.asarray(opening, float)
    opening -= (opening @ approach) * approach
    if np.linalg.norm(opening) < 1e-6:
        return None
    opening /= np.linalg.norm(opening)
    transverse = np.cross(opening, approach)
    rotation = np.column_stack((transverse, opening, approach))
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or
            not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)):
        return None
    return rotation


def geometric_candidates(observation, points_base, *, table_height_m,
                         max_opening_m=MAX_OPENING_M, opening_margin_m=OPENING_MARGIN_M,
                         limit=12, max_raw=240, approach_distances=(0.08, 0.05),
                         azimuths=8, oblique_tilt_deg=(0.0, 30.0, 55.0),
                         band=(0.25, 0.75), seed=0, hull_vertices=None,
                         T_tip_tcp=None):
    """Antipodal grasp hypotheses from observed geometry only.

    Uses the observation cloud and supported height; no colour, no shape
    class, no network. Grip centres are sampled in the mid-height band, each
    tried against a ring of horizontal and oblique approach directions --
    steep tilts matter for low objects where a horizontal wrist would hit
    the support plane. Width comes from the cloud (and hull when supplied),
    and every hypothesis passes the same ``revalidate_candidate`` as network
    candidates. The returned list is diversified across tilt classes instead
    of letting one geometry dominate the ranking.
    """
    points = np.asarray(points_base, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 100:
        return []
    height = float(observation.get('dimensions_m', [0, 0, 0])[2] or 0)
    center = np.asarray(observation.get('obb_center_base_m'), dtype=float)
    if height <= 0 or center.shape != (3,) or not np.all(np.isfinite(center)):
        return []
    z_lo = table_height_m + band[0] * height
    z_hi = table_height_m + band[1] * height
    band_points = points[(points[:, 2] >= z_lo) & (points[:, 2] <= z_hi)]
    if len(band_points) < 20:
        return []
    rng = np.random.default_rng(seed)
    if len(band_points) > max_raw:
        band_points = band_points[rng.choice(len(band_points), max_raw, replace=False)]

    tilt_axes = []
    for azimuth in range(azimuths):
        for tilt in oblique_tilt_deg:
            tilt_axes.append((2 * math.pi * azimuth / azimuths, math.radians(tilt)))
    raw = []
    for grip in band_points:
        for azimuth, tilt in tilt_axes:
            horizontal = np.array([math.cos(azimuth), math.sin(azimuth), 0.0])
            approach = (math.cos(tilt) * horizontal
                        + math.sin(tilt) * np.array([0.0, 0.0, -1.0]))
            opening = np.cross(np.array([0.0, 0.0, 1.0]), horizontal)
            rotation = _approach_rotation(approach, opening)
            if rotation is None:
                continue
            tip_position = np.array([center[0], center[1], grip[2]]) \
                if tilt == 0.0 else grip.copy()
            tip = np.eye(4)
            tip[:3, :3] = rotation
            tip[:3, 3] = tip_position
            # Local width across the candidate opening axis decides feasibility
            # before the shared revalidation runs.
            offsets = points - tip_position
            y = offsets @ rotation[:, OPENING_AXIS]
            z = offsets @ rotation[:, APPROACH_AXIS]
            near = (np.abs(z) <= GRIP_REGION_LENGTH_M) & (np.abs(offsets @ rotation[:, TRANSVERSE_AXIS]) <= 0.03)
            if int(near.sum()) < 6:
                continue
            width = float(np.ptp(np.quantile(y[near], WIDTH_ROBUST_Q)))
            if hull_vertices is not None and len(hull_vertices) >= 4:
                hull_offsets = np.asarray(hull_vertices, dtype=float) - tip_position
                hull_window = ((np.abs(hull_offsets @ rotation[:, APPROACH_AXIS]) <= 0.03) &
                               (np.abs(hull_offsets @ rotation[:, TRANSVERSE_AXIS]) <= 0.04))
                if int(hull_window.sum()) >= 3:
                    width = max(width, float(np.ptp(np.quantile(
                        hull_offsets[hull_window] @ rotation[:, OPENING_AXIS],
                        WIDTH_ROBUST_Q))))
            opening_width = width + opening_margin_m
            if not 0.006 <= opening_width <= max_opening_m:
                continue
            for distance in approach_distances:
                pregrasp_tip = tip.copy()
                pregrasp_tip[:3, 3] = tip_position - distance * approach
                # The flange (tcp_link) trails the fingertip midpoint by the
                # model tip offset; candidates must carry that transform like
                # network candidates do, not identify TCP with the tip.
                if T_tip_tcp is not None:
                    tip_matrix = np.asarray(T_tip_tcp, dtype=float)
                    tcp = tip @ tip_matrix
                    pregrasp = pregrasp_tip @ tip_matrix
                    tcp_ok = bool(np.allclose(tcp[3], [0, 0, 0, 1], atol=1e-6) and
                                  np.isclose(np.linalg.det(tcp[:3, :3]), 1.0, atol=1e-5))
                else:
                    tcp, pregrasp, tcp_ok = tip, pregrasp_tip, False
                raw.append({
                    'source': 'geometric_antipodal', 'method': 'geometry',
                    'frame': TIP_FRAME_TO_WORLD,
                    'stamp': dict(observation.get('stamp') or {}),
                    'T_base_tip': tip.tolist(), 'T_base_tcp': tcp.tolist(),
                    'T_base_pregrasp_tcp': pregrasp.tolist(),
                    'tcp_transform_applied': tcp_ok,
                    'opening_m': opening_width, 'predicted_width_m': width,
                    'approach_direction': approach.tolist(),
                    'approach_distance_m': float(distance),
                    'score': None, 'grip_height_m': float(grip[2]),
                    'raw_azimuth_deg': math.degrees(azimuth), 'raw_tilt_deg': math.degrees(tilt),
                })
    if not raw:
        return []
    dedup = {}
    for candidate in raw:
        key = (round(candidate['T_base_tip'][0][3], 4), round(candidate['T_base_tip'][1][3], 4),
               round(candidate['T_base_tip'][2][3], 4),
               tuple(round(v, 2) for v in np.asarray(candidate['approach_direction'])),
               candidate['approach_distance_m'])
        dedup.setdefault(key, candidate)
    annotated = apply_revalidation(list(dedup.values()), points,
                                   hull_vertices=hull_vertices,
                                   table_height_m=table_height_m,
                                   max_opening_m=max_opening_m)
    valid = [c for c in annotated if c['rejection_reason'] is None]
    # Diversify: round-robin over tilt classes so one comfortable geometry
    # cannot crowd out steeper approaches that low objects need.
    by_tilt = {}
    for candidate in sorted(valid, key=lambda c: -c['confidence']):
        by_tilt.setdefault(candidate['raw_tilt_deg'], []).append(candidate)
    diversified, classes = [], list(by_tilt)
    while len(diversified) < limit and any(by_tilt[c] for c in classes):
        for tilt in classes:
            if by_tilt[tilt]:
                diversified.append(by_tilt[tilt].pop(0))
                if len(diversified) == limit:
                    break
    for candidate_id, candidate in enumerate(diversified):
        candidate['id'] = candidate_id
    return diversified
