import argparse
import hashlib
import shutil
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from data import DATASET, REVISION, check_layout, write_json


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description='Download the complete official HW2 dataset')
    p.add_argument('--raw', default='data/raw')
    p.add_argument('--reports', default='reports')
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    raw = Path(args.raw).resolve()
    raw.mkdir(parents=True, exist_ok=True)
    info = HfApi().dataset_info(DATASET, revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        raise ValueError('Dataset revision does not match')
    expected_bytes = sum(f.size or 0 for f in info.siblings)
    local_bytes = sum(
        min((raw / f.rfilename).stat().st_size, f.size or 0)
        for f in info.siblings if (raw / f.rfilename).is_file()
    )
    required = max(0, expected_bytes - local_bytes) + 512 * 1024**2
    if shutil.disk_usage(raw).free < required:
        raise RuntimeError(f'Not enough disk space. Need about {required / 1024**3:.2f} GiB free')
    print(f'Dataset: {DATASET}\nRevision: {REVISION}\nLocal folder: {raw}', flush=True)
    print(f'Repository size: {expected_bytes / 10**9:.2f} GB', flush=True)
    snapshot_download(
        repo_id=DATASET, repo_type='dataset', revision=REVISION,
        local_dir=str(raw), max_workers=args.workers,
    )
    manifest = []
    for item in info.siblings:
        path = raw / item.rfilename
        if not path.is_file() or (item.size is not None and path.stat().st_size != item.size):
            raise RuntimeError(f'Missing or incomplete file: {path}')
        digest = sha256_file(path)
        lfs = item.lfs
        expected_sha = (lfs.get('sha256') if isinstance(lfs, dict)
                        else getattr(lfs, 'sha256', None)) if lfs else None
        if expected_sha and digest != expected_sha:
            raise RuntimeError(f'SHA256 mismatch: {path}. File was not modified by this script.')
        manifest.append({'file': item.rfilename, 'bytes': path.stat().st_size,
                         'sha256': digest, 'remote_sha256_verified': bool(expected_sha)})
    sizes = check_layout(raw)
    report = {'dataset': DATASET, 'revision': REVISION, 'raw': str(raw),
              'counts': sizes, 'files': manifest, 'status': 'downloaded and verified'}
    write_json(Path(args.reports) / 'download_manifest.json', report)
    print(f'COMPLETE: train={sizes["train"]:,}, test={sizes["test"]:,}', flush=True)


if __name__ == '__main__':
    main()
