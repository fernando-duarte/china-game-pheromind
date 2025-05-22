import os
from china_data.utils import get_project_root, find_file, ensure_directory, get_output_directory
from china_data.utils.path_constants import PACKAGE_DIR_NAME, OUTPUT_DIR_NAME


def test_get_project_root_returns_existing_directory():
    root = get_project_root()
    assert os.path.isdir(root)
    # Check that the directory exists and contains the china_data directory
    assert os.path.isdir(os.path.join(root, 'china_data'))


def test_find_file_locates_known_file():
    path = find_file('README.md', [PACKAGE_DIR_NAME])
    assert path and path.endswith(os.path.join(PACKAGE_DIR_NAME, 'README.md'))


def test_ensure_directory_creates_path(tmp_path):
    new_dir = tmp_path / 'sub'
    path = ensure_directory(str(new_dir))
    assert os.path.isdir(path)


def test_get_output_directory_exists():
    out_dir = get_output_directory()
    expected = os.path.join(get_project_root(), PACKAGE_DIR_NAME, OUTPUT_DIR_NAME)
    assert out_dir == expected
    assert os.path.isdir(out_dir)

