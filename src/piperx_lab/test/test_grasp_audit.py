from copy import deepcopy
from piperx_lab.grasp_audit import audit_preview


def example():
    names = [f'joint{i}' for i in range(1, 7)]
    return {'status': 'preview_path_found', 'observation': {'stamp': {'sec': 100, 'nanosec': 0}},
            'initial_real_joint_state': {'names': names, 'positions': [0.] * 6},
            'trajectories': [{'joint_names': names, 'points': [
                {'positions': [value] * 6, 'time_from_start_s': seconds}
                for value, seconds in [(start, 0.), (end, 1.)]]}
                for start, end in [(0., .05), (.05, .1)]]}


def test_record_checks_do_not_grant_execution():
    result = audit_preview(example(), now=100.2)
    assert result['record_checks_pass']
    assert result['execution_ready'] is False
    assert not audit_preview(example(), now=102)['record_checks_pass']
    assert not audit_preview(example(), now=99)['record_checks_pass']


def test_segment_boundary_jump_and_invalid_record():
    record = example()
    record['trajectories'][1]['points'][0]['positions'][0] = 1.
    assert not audit_preview(record, now=100.2)['record_checks_pass']
    for change in ('missing', 'nan', 'time'):
        record = deepcopy(example())
        if change == 'missing':
            record['trajectories'][0]['joint_names'] = ['joint1']
        elif change == 'nan':
            record['trajectories'][0]['points'][0]['positions'][0] = float('nan')
        else:
            record['trajectories'][0]['points'][1]['time_from_start_s'] = -1.
        assert not audit_preview(record, now=100.2)['record_checks_pass']
