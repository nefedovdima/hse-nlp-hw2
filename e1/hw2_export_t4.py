"""Run ON Vast.ai after epoch 1 to build a portable Colab T4 inference bundle.

No additional training data, labels or target generation. Never changes checkpoint.
"""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile

import pyarrow as pa
import pyarrow.ipc as ipc
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'exports'
CACHE = ROOT / 'data/prepared/bpe8192'
CKPT = ROOT / 'checkpoints/e01_last.pt'
TOKENIZER = CACHE / 'tokenizer.json'
SOURCE = Path(__file__).parent / 'hw2_model.py'
TEST = CACHE / 'test.arrow'
EXPECTED_TOKENIZER_SHA = '659f7740d5fc78e8ad7c7b395331fc294a6186f270b6643272dce80e640fc6d2'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        for b in iter(lambda: file.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    for p in (CKPT, TOKENIZER, SOURCE, TEST):
        if not p.is_file():
            raise FileNotFoundError(p)
    assert sha256(TOKENIZER) == EXPECTED_TOKENIZER_SHA, 'Unexpected tokenizer!'

    OUT.mkdir(exist_ok=True, parents=True)
    state = torch.load(CKPT, map_location='cpu', weights_only=False)
    assert state['step'] == 57502, f"Expected final first epoch: {state['step']}"

    original = state['model']
    assert torch.equal(original['head.weight'], original['emb.weight']), 'Weights are not tied'
    # Head is tied to the embedding in CodeTransformer.__init__; do not store twice.
    weights = {key: value.detach().to(device='cpu', dtype=torch.float16).contiguous()
               for key, value in original.items() if key != 'head.weight'}
    model_data = {
        'step': state['step'],
        'model_config': state['model_config'],
        'model': weights,
        'precision': 'fp16',
    }
    model_file = OUT / 'model_fp16.pt'
    torch.save(model_data, model_file)
    del model_data, state, original, weights

    with pa.memory_map(str(TEST), 'r') as memory:
        reader = ipc.open_file(memory)
        rows = []
        for index in range(reader.num_record_batches):
            batch = reader.get_batch(index).select(['row_id', 'input_ids'])
            rows.extend(batch.to_pylist())

    rows.sort(key=lambda r: r['row_id'])
    ids = [r['row_id'] for r in rows]
    assert len(rows) == 1919 and len(set(ids)) == 1919, 'Unexpected test count / duplicate row_ids'
    assert all(len(r['input_ids']) > 0 and r['input_ids'][-1] == 1 for r in rows), 'EOS missing'
    assert all(all(0 <= int(t) < 8192 for t in r['input_ids']) for r in rows), 'Unexpected token'
    test_file = OUT / 'test_inputs.json'
    test_file.write_text(json.dumps(rows, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    shutil.copy2(TOKENIZER, OUT / 'tokenizer.json')
    shutil.copy2(SOURCE, OUT / 'hw2_model.py')

    manifest = {
        'step': 57502,
        'test_rows': len(rows),
        'model_config': {'name': 'Transformer encoder-decoder', **{
            k: v for k, v in torch.load(model_file, map_location='cpu', weights_only=True)['model_config'].items()}},
        'tokenizer_sha256': EXPECTED_TOKENIZER_SHA,
        'format': 'original test row_id + token ids (EOS included); output by row_id',
        'sha256': {p.name: sha256(p) for p in
                   (model_file, test_file, OUT / 'tokenizer.json', OUT / 'hw2_model.py')},
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    archive = OUT / 'hw2_t4_bundle.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as z:
        for fname in ('model_fp16.pt', 'test_inputs.json', 'tokenizer.json', 'hw2_model.py', 'manifest.json'):
            z.write(OUT / fname, fname)

    print('EXPORTED:', archive)
    print('Archive size MiB:', round(archive.stat().st_size / (1024**2), 2))
    print('Checkpoint step:', manifest['step'])
    print('Test examples:', len(rows))
    print('Tokenizer sha256:', EXPECTED_TOKENIZER_SHA)
    print('First 5 row IDs:', ids[:5])
    print('Model sha256:', manifest['sha256']['model_fp16.pt'])
    print('Original checkpoint not modified.')


if __name__ == '__main__':
    main()
