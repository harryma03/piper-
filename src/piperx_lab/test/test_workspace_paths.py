"""Guard the paths needed after cloning this workspace elsewhere."""

import json
from pathlib import Path

from piperx_lab.workspace import workspace_root


ROOT = Path(__file__).resolve().parents[3]


def test_workspace_override_uses_checkout_path(monkeypatch, tmp_path):
    (tmp_path / 'calibration').mkdir()
    (tmp_path / 'calibration/board.json').write_text('{}')
    (tmp_path / 'src/piperx_lab').mkdir(parents=True)
    (tmp_path / 'src/piperx_lab/package.xml').write_text('<package/>')
    monkeypatch.setenv('PIPER_LAB_WORKSPACE', str(tmp_path))
    monkeypatch.chdir(ROOT / 'src/piperx_lab')
    assert workspace_root() == tmp_path


def test_current_samples_reference_files_relatively():
    for directory in ('calibration/samples', 'records/gazebo_piper_handeye/samples'):
        samples = sorted((ROOT / directory).glob('*.json'))
        assert len(samples) == 12
        for path in samples:
            sample = json.loads(path.read_text(encoding='utf-8'))
            for key in ('board_json', 'color_image', 'detected_image'):
                reference = Path(sample[key])
                assert not reference.is_absolute(), (path, key)
                assert (path.parent / reference).is_file(), (path, key)


def test_installed_board_copy_matches_capture_board():
    for filename in ('board.json', 'board.png'):
        assert ((ROOT / 'calibration' / filename).read_bytes()
                == (ROOT / 'src/piperx_lab/calibration' / filename).read_bytes())
