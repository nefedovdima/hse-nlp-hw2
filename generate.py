"""Run the original V2 T4 benchmark and check against the submitted answers."""
import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BUNDLE_SHA256 = '48bf17d252bad5b34acba190f5d895c20da9c19a9ffc13f31e15b2c50c08663a'
MANIFEST_SHA256 = '9217a4bf527eea843dbc945ab652c9d4fbefde7e82008e3fa7dc5f78793188df'
SUBMISSION_SHA256 = 'e3b2f9c9170111dc17a17800c992c72c2250337b4c7e5c8a93f87cb7b309e9f5'
TOKENIZER_SHA256 = '659f7740d5fc78e8ad7c7b395331fc294a6186f270b6643272dce80e640fc6d2'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_submission(path):
    lines = Path(path).read_text(encoding='utf-8').splitlines()
    if len(lines) != 1919:
        raise ValueError(f'Expected 1919 JSONL lines; found {len(lines)}')
    rows = [json.loads(line) for line in lines]
    if any(not isinstance(row, dict) or set(row) != {'code'}
           or not isinstance(row['code'], str) for row in rows):
        raise ValueError('Each line must be an object containing only a string code')
    return rows


def prepare_bundle(destination):
    archive = ROOT / 'model_weights/t4_step111275.zip'
    if sha256(archive) != BUNDLE_SHA256:
        raise ValueError('The trained model archive has changed')
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        expected = {'03_inference_T4_v2.ipynb', 'hw2_common_v2.py',
                    'hw2_infer_t4_v2.py', 'hw2_model_v2.py', 'manifest.json',
                    'model_fp16.pt', 'test_inputs.json', 'tokenizer.json'}
        if set(bundle.namelist()) != expected:
            raise ValueError('Unexpected model archive contents')
        bundle.extractall(destination)
    if sha256(destination / 'manifest.json') != MANIFEST_SHA256:
        raise ValueError('The original model manifest has changed')
    manifest = json.loads((destination / 'manifest.json').read_text(encoding='utf-8'))
    for name, expected_hash in manifest['files'].items():
        if Path(name).name != name or sha256(destination / name) != expected_hash:
            raise ValueError(f'Model bundle hash mismatch: {name}')
    if manifest['checkpoint_step'] != 111275 or manifest['rows'] != 1919:
        raise ValueError('Unexpected checkpoint or test set')
    if sha256(destination / 'tokenizer.json') != TOKENIZER_SHA256:
        raise ValueError('Tokenizer hash mismatch')
    rows = json.loads((destination / 'test_inputs.json').read_text(encoding='utf-8'))
    if [row['row_id'] for row in rows] != list(range(1919)):
        raise ValueError('Missing, duplicate or unordered test examples')
    if any(not row['input_ids'] or row['input_ids'][-1] != 1
           or len(row['input_ids']) > 2048
           or any(type(token) is not int or not 0 <= token < 8192 for token in row['input_ids'])
           for row in rows):
        raise ValueError('Invalid test tokens')
    return destination


def check_reference():
    reference = ROOT / 'submission.jsonl'
    validate_submission(reference)
    if sha256(reference) != SUBMISSION_SHA256:
        raise ValueError('The submitted reference file has changed')
    original = json.loads((ROOT / 'evidence/t4_benchmark.json').read_text(encoding='utf-8'))
    if (original['bundle_manifest_sha256'] != MANIFEST_SHA256
            or original['predictions_sha256'] != [SUBMISSION_SHA256] * 3):
        raise ValueError('The recorded T4 benchmark does not match this model and submission')
    return reference


def compare_submission(actual, reference):
    rows = validate_submission(actual)
    if Path(actual).read_bytes() != Path(reference).read_bytes():
        reference_rows = validate_submission(reference)
        changed = [i for i, (a, b) in enumerate(zip(rows, reference_rows)) if a != b]
        detail = f'First changed row IDs: {changed[:10]}' if changed else 'JSONL serialization differs'
        raise RuntimeError(f'Generated submission differs from the submitted reference. {detail}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--check-only', action='store_true', help='Check files without loading PyTorch or generating')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    output = args.output.resolve()
    reference = check_reference()
    bundle = prepare_bundle(output / 'bundle')
    print('Model files, tokenizer, test order and reference submission verified.', flush=True)
    if args.check_only:
        return
    # Run in a separate process so imports resolve to the untouched bundle.
    subprocess.run([
        sys.executable, str(bundle / 'hw2_infer_t4_v2.py'),
        '--bundle', str(bundle), '--batch-size', '32', '--max-new-tokens', '384',
        '--repeats', str(args.repeats), '--output', str(output),
    ], check=True, cwd=bundle)
    for run in range(1, args.repeats + 1):
        compare_submission(output / f'predictions_run{run}.jsonl', reference)
    compare_submission(output / 'submission.jsonl', reference)
    print(f'All {args.repeats} runs match the submitted file byte for byte.', flush=True)
    print(f'Submission: {output / "submission.jsonl"}', flush=True)


if __name__ == '__main__':
    main()
