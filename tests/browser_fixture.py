"""Create a fresh disposable one-frame dataset for the browser smoke test."""
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from labeling.app import Dataset, index_images, import_legacy

if __name__ == '__main__':
    repo = Path(__file__).resolve().parents[1]
    root = Path(tempfile.mkdtemp(prefix='homography-phases-'))
    (root / 'images').mkdir()
    (root / 'labels').mkdir()
    shutil.copyfile(repo / 'frames/601.png', root / 'images/601.png')
    shutil.copyfile(repo / 'labels/601.png.json', root / 'labels/601.png.json')
    index_images(root / 'images', root / 'dataset', 'smoke')
    import_legacy(Dataset(root / 'dataset'), root / 'labels', root / 'images')
    print(root / 'dataset')
