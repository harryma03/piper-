"""Find this project's data directory after the checkout has been moved."""

import os
from pathlib import Path


def _is_workspace(path: Path) -> bool:
    return ((path / 'calibration/board.json').is_file()
            and (path / 'src/piperx_lab/package.xml').is_file())


def workspace_root() -> Path:
    """Locate the checkout without relying on a particular username or cwd.

    A regular ROS install can set PIPER_LAB_WORKSPACE if the source checkout is
    elsewhere. In a symlink install, this module itself points back to source.
    """
    override = os.environ.get('PIPER_LAB_WORKSPACE')
    if override:
        path = Path(override).expanduser().resolve()
        if not _is_workspace(path):
            raise FileNotFoundError(
                f'PIPER_LAB_WORKSPACE is not a Piper checkout: {path}')
        return path

    starts = [Path.cwd().resolve(), Path(__file__).resolve()]
    try:
        from ament_index_python.packages import get_package_share_directory
        starts.append(Path(get_package_share_directory('piperx_lab')).resolve())
    except (ImportError, LookupError):
        pass
    for start in starts:
        for candidate in (start, *start.parents):
            if _is_workspace(candidate):
                return candidate
    raise FileNotFoundError(
        'Piper checkout not found; set PIPER_LAB_WORKSPACE to its root directory')
