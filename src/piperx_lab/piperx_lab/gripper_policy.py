"""Adaptive gripper policy and grasp-state classification.

Replaces the fixed 32 mm close command. Closing targets, pre-opening and the
grasp-state interpretation all derive from the candidate's predicted width
and the robot configuration -- no per-shape hard-coding. The simulated
fingers are position controlled (each finger follows width/2), so a command
below the object width produces contact force through the position error;
the squeeze must stay small because force grows with the commanded error.
"""
import math

# Simulated Piper gripper envelope (target widths, not measured forces).
MAX_OPENING_M = 0.070
MIN_CLOSE_WIDTH_M = 0.008
DEFAULT_SQUEEZE_M = 0.006        # fallback undershoot when width is unknown
SQUEEZE_WIDTH_FRACTION = 0.10    # ~0.36 N/side at kp=300 for a 24 mm part
MIN_SQUEEZE_M = 0.002
MAX_SQUEEZE_M = 0.008
PRE_OPEN_MARGIN_M = 0.010        # extra clearance when opening before approach
FINGER_JOINTS = ("gripper", "gripper_joint1", "gripper_joint2")


def pre_open_width(candidate, *, max_opening_m=MAX_OPENING_M,
                   margin_m=PRE_OPEN_MARGIN_M):
    """Width commanded before the approach: candidate opening plus clearance."""
    opening = float(candidate.get('opening_m', max_opening_m) or max_opening_m)
    predicted = candidate.get('predicted_width_m')
    if isinstance(predicted, (int, float)) and math.isfinite(predicted):
        opening = max(opening, float(predicted) + margin_m)
    return float(min(max(opening, 0.01), max_opening_m))


def close_width_for(candidate, *, squeeze_m=None, min_width_m=MIN_CLOSE_WIDTH_M,
                    max_opening_m=MAX_OPENING_M):
    """Commanded closing width from the predicted local object width.

    ``predicted_width_m`` is the observed object width along the candidate's
    opening axis (revalidated against the cloud). The command undershoots it
    so both fingers press on the object. The squeeze scales with the object:
    with position-controlled fingers the contact force grows with the
    commanded undershoot, and a fixed 6 mm squeeze over-presses small parts
    hard enough to push the wrist joints out of tolerance. A fixed override
    is still accepted through ``squeeze_m``; unknown widths fall back to the
    candidate opening.
    """
    predicted = candidate.get('predicted_width_m')
    if isinstance(predicted, (int, float)) and math.isfinite(predicted) and predicted > 0:
        if squeeze_m is None:
            squeeze = min(max(SQUEEZE_WIDTH_FRACTION * float(predicted),
                              MIN_SQUEEZE_M), MAX_SQUEEZE_M)
        else:
            squeeze = float(squeeze_m)
        target = float(predicted) - squeeze
    else:
        opening = float(candidate.get('opening_m', max_opening_m) or max_opening_m)
        target = min(opening, max_opening_m) - float(squeeze_m if squeeze_m is not None
                                                     else DEFAULT_SQUEEZE_M)
    return float(min(max(target, min_width_m), max_opening_m))


def closing_sequence(predicted_width, *, max_opening_m=MAX_OPENING_M,
                     min_width_m=MIN_CLOSE_WIDTH_M, step_m=0.002,
                     max_steps=8, approach_margin_m=0.004):
    """Widths to command while closing until both fingers make contact.

    Starts just outside the predicted object width and steps inward; the
    caller stops as soon as both finger contact sensors fire, so the grip
    force comes from the smallest sufficient undershoot instead of a fixed
    position overshoot. The final entry is the floor ``min_width_m``.
    """
    if not (isinstance(predicted_width, (int, float))
            and math.isfinite(predicted_width) and predicted_width > 0):
        return [max_opening_m * 0.5, min_width_m]
    widths = []
    width = min(float(predicted_width) + approach_margin_m, max_opening_m)
    floor = max(min_width_m, float(predicted_width) - step_m * max_steps)
    while width > floor and len(widths) < max_steps:
        widths.append(round(width, 5))
        width -= step_m
    widths.append(floor)
    return widths


def classify_grasp(contacts, finger_joints_end, commanded_close_width, *,
                   lift_m, min_lift_m=0.07, contact_error_m=0.004,
                   both_phases=False):
    """Interpret closing/lifting evidence into a grasp state.

    ``contacts`` maps phase -> finger -> bool (as collected from the finger
    bumper topics). ``finger_joints_end`` is the final finger joint vector
    (gripper_joint1, gripper_joint2) or None.

    States: object_grasped, contact_without_lift (both fingers pressed in
    both phases but the object never rose), slipped (contact during closing
    lost while lifting), single_sided, empty_grip, blocked_unknown (finger
    stopped far short of the command without registered contact).
    """
    closing = contacts.get('closing', {})
    lifting = contacts.get('lifting', {})
    both_closing = bool(closing.get('gripper_link1') and closing.get('gripper_link2'))
    both_lifting = bool(lifting.get('gripper_link1') and lifting.get('gripper_link2'))
    any_closing = bool(closing.get('gripper_link1') or closing.get('gripper_link2'))
    any_lifting = bool(lifting.get('gripper_link1') or lifting.get('gripper_link2'))
    lifted = lift_m is not None and lift_m >= min_lift_m

    details = {
        'both_fingers_closing': both_closing,
        'both_fingers_lifting': both_lifting,
        'lift_m': lift_m,
        'commanded_close_width_m': commanded_close_width,
    }
    if both_phases:
        details['both_fingers_both_phases'] = both_closing and both_lifting
    if both_closing and both_lifting:
        # Both fingers pressed through both phases, but a near-zero lift
        # means the object was pressed against the support plane rather
        # than carried; that distinction belongs in the label.
        return ('object_grasped' if lifted else 'contact_without_lift'), details
    if (any_closing or any_lifting) and not both_closing and not both_lifting:
        return 'single_sided', details
    if both_closing and not both_lifting:
        return 'slipped', details
    if not any_closing and not any_lifting:
        # Fingers stopped far short of the command means something blocked
        # them without a registered bumper contact.
        if finger_joints_end is not None and len(finger_joints_end) == 2:
            reached_width = abs(finger_joints_end[0] - finger_joints_end[1])
            details['reached_width_m'] = float(reached_width)
            if commanded_close_width - reached_width > contact_error_m:
                return 'blocked_unknown', details
        return 'empty_grip', details
    if both_closing and both_lifting and not lifted:
        return 'object_grasped', details
    return 'blocked_unknown', details
