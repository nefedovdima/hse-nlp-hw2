import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

DATASET = 'SlavaYus/tinysms_qa_12m'
REVISION = 'b2724689ff079e9c22efd548fe134bfb29036dbb'
TRAIN_ROWS = 11_838_003
TEST_ROWS = 1_919
TRAIN_FILES = 60


def is_validation(question):
    key = ' '.join(question.split()).casefold().encode('utf-8')
    return int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') % 100 < 5


def train_files(raw):
    return sorted((Path(raw) / 'data' / 'train').glob('part-*.parquet'))


def iter_rows(files, columns=('question', 'code'), batch_size=4096):
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=batch_size, columns=list(columns)
        ):
            yield from batch.to_pylist()


def iter_split(raw, split='train'):
    if split not in ('train', 'validation'):
        raise ValueError('split must be train or validation')
    want_val = split == 'validation'
    for row in iter_rows(train_files(raw)):
        if any(not isinstance(row[k], str) or not row[k].strip()
               for k in ['question', 'code']):
            continue
        if is_validation(row['question']) == want_val:
            yield row


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def check_layout(raw, expected_train=TRAIN_ROWS, expected_test=TEST_ROWS,
                 expected_files=TRAIN_FILES):
    raw = Path(raw)
    paths = train_files(raw)
    if len(paths) != expected_files:
        raise ValueError(f'Expected {expected_files} train files, found {len(paths)}')
    test = raw / 'data' / 'test' / 'test.parquet'
    sizes = {'train': 0, 'test': 0}
    for split, files in [('train', paths), ('test', [test])]:
        for path in files:
            f = pq.ParquetFile(path)
            expected = ['question', 'code'] if split == 'train' else ['question']
            if sorted(f.schema_arrow.names) != sorted(expected):
                raise ValueError(f'Unexpected schema: {path}, {f.schema_arrow}')
            sizes[split] += f.metadata.num_rows
    if sizes != {'train': expected_train, 'test': expected_test}:
        raise ValueError(f'Unexpected row counts: {sizes}')
    return sizes
