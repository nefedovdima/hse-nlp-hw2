"""Train a byte-level BPE on the local training split; encode every official row."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault('RAYON_NUM_THREADS', '16')

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from audit_full import quantiles
from data import REVISION, check_layout, is_validation, train_files, write_json
from download_all import sha256_file

SPECIAL = ['<pad>', '<eos>', '<unk>', '<bos>']


def batches(path, batch_size=4096):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
        yield batch.to_pylist()


def valid(row, test=False):
    keys = ['question'] if test else ['question', 'code']
    return all(isinstance(row.get(k), str) and row[k].strip() for k in keys)


def fit_tokenizer(paths, destination, signature, vocab_size=8192):
    destination = Path(destination)
    meta = destination.with_suffix('.meta.json')
    if destination.exists() and meta.exists():
        info = json.loads(meta.read_text())
        if info['signature'] != signature or info['sha256'] != sha256_file(destination):
            raise ValueError('Existing tokenizer does not match data/settings; use another --out folder')
        print('Reusing the completed tokenizer', flush=True)
        return Tokenizer.from_file(str(destination)), info

    counts = {'train_rows_used': 0, 'validation_rows_excluded': 0, 'invalid_rows': 0}

    def texts():
        for i, path in enumerate(paths, 1):
            for rows in batches(path):
                for row in rows:
                    if not valid(row):
                        counts['invalid_rows'] += 1
                    elif is_validation(row['question']):
                        counts['validation_rows_excluded'] += 1
                    else:
                        counts['train_rows_used'] += 1
                        yield row['question']
                        yield row['code']
            print(f'BPE corpus [{i}/{len(paths)}]: {counts["train_rows_used"]:,} training rows', flush=True)

    tok = Tokenizer(models.BPE(unk_token='<unk>'))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, min_frequency=2, special_tokens=SPECIAL,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=True,
    )
    start = time.perf_counter()
    print('Training BPE on ALL local train rows, excluding held-out validation...', flush=True)
    tok.train_from_iterator(texts(), trainer=trainer)
    if not counts['train_rows_used']:
        raise ValueError('No training rows for tokenizer')
    if tok.get_vocab_size() > 65535:
        raise ValueError('Vocabulary exceeds uint16 cache format')
    tmp = destination.with_suffix('.tmp.json')
    tok.save(str(tmp))
    tmp.replace(destination)
    info = {'signature': signature, 'sha256': sha256_file(destination),
            'vocab_size': tok.get_vocab_size(), 'counts': counts,
            'special_tokens': {s: tok.token_to_id(s) for s in SPECIAL},
            'elapsed_seconds': round(time.perf_counter() - start, 2)}
    write_json(meta, info)
    return tok, info


def encode_shard(path, dest, tok, tokenizer_sha, row_start=0, test=False,
                 batch_size=4096, check_first=32):
    path, dest = Path(path), Path(dest)
    source_sha = sha256_file(path)
    meta = dest.with_suffix('.meta.json')
    sig = {'source_sha256': source_sha, 'tokenizer_sha256': tokenizer_sha,
           'row_start': row_start, 'test': test, 'format': 1}
    if dest.exists() and meta.exists():
        result = json.loads(meta.read_text())
        if (result.get('signature') == sig and
                result.get('cache_sha256') == sha256_file(dest)):
            print(f'Reusing {dest.name}', flush=True)
            return result

    fields = [('row_id', pa.int64()), ('input_ids', pa.list_(pa.uint16())),
              ('input_length', pa.int32())]
    if not test:
        fields += [('labels', pa.list_(pa.uint16())), ('target_length', pa.int32()),
                   ('is_validation', pa.bool_())]
    schema = pa.schema(fields)
    eos = tok.token_to_id('<eos>')
    unk = tok.token_to_id('<unk>')
    result = {'signature': sig, 'rows': 0, 'train_rows': 0, 'validation_rows': 0,
              'invalid_rows': 0, 'input_tokens': 0, 'target_tokens': 0,
              'input_histogram': {}, 'target_histogram': {},
              'roundtrip_texts_checked': 0, 'unknown_tokens': 0}
    tmp = dest.with_suffix('.arrow.tmp')
    dest.parent.mkdir(parents=True, exist_ok=True)
    seen = 0
    with pa.OSFile(str(tmp), 'wb') as sink, ipc.new_file(sink, schema) as writer:
        for rows in batches(path, batch_size):
            ids = list(range(row_start + seen, row_start + seen + len(rows)))
            seen += len(rows)
            if any(not valid(r, test=test) for r in rows):
                # Keep row identity unambiguous; the full audit found zero such rows.
                raise ValueError(f'Empty/nonstring row in {path}; no examples have been silently dropped')
            texts = [r['question'] for r in rows]
            if not test:
                texts += [r['code'] for r in rows]
            encoded = tok.encode_batch(texts, add_special_tokens=False)
            if not result['roundtrip_texts_checked']:
                for j in range(min(check_first, len(rows))):
                    for index in ([j] if test else [j, len(rows) + j]):
                        if tok.decode(encoded[index].ids, skip_special_tokens=False) != texts[index]:
                            raise ValueError(f'Tokenizer failed exact roundtrip in {path}')
                        result['roundtrip_texts_checked'] += 1
            result['unknown_tokens'] += sum(enc.ids.count(unk) for enc in encoded)
            x = [e.ids + [eos] for e in encoded[:len(rows)]]
            lx = [len(v) for v in x]
            columns = [pa.array(ids, type=pa.int64()),
                       pa.array(x, type=pa.list_(pa.uint16())),
                       pa.array(lx, type=pa.int32())]
            if not test:
                y = [e.ids + [eos] for e in encoded[len(rows):]]
                ly = [len(v) for v in y]
                val = [is_validation(r['question']) for r in rows]
                columns += [pa.array(y, type=pa.list_(pa.uint16())),
                            pa.array(ly, type=pa.int32()), pa.array(val, type=pa.bool_())]
                result['validation_rows'] += sum(val)
                result['train_rows'] += len(rows) - sum(val)
                result['target_tokens'] += sum(ly)
                for n in ly:
                    result['target_histogram'][str(n)] = result['target_histogram'].get(str(n), 0) + 1
            writer.write_batch(pa.RecordBatch.from_arrays(columns, schema=schema))
            result['rows'] += len(rows)
            result['input_tokens'] += sum(lx)
            for n in lx:
                result['input_histogram'][str(n)] = result['input_histogram'].get(str(n), 0) + 1
    if result['unknown_tokens']:
        raise ValueError(f'Unknown tokens detected in {path}')
    tmp.replace(dest)
    result['cache_sha256'] = sha256_file(dest)
    result['bytes'] = dest.stat().st_size
    write_json(meta, result)
    return result


def summarize(results):
    summary = {key: sum(r[key] for r in results) for key in
               ['rows', 'train_rows', 'validation_rows', 'input_tokens', 'target_tokens',
                'roundtrip_texts_checked', 'unknown_tokens', 'bytes']}
    for key in ['input', 'target']:
        hist = {}
        for r in results:
            for length, count in r[key + '_histogram'].items():
                n = int(length)
                hist[n] = hist.get(n, 0) + count
        summary[key + '_length_quantiles'] = quantiles(hist)
        summary[key + '_rows_over_limit'] = {
            str(limit): sum(c for n, c in hist.items() if n > limit)
            for limit in [128, 192, 256, 384, 512, 768, 1024]
        }
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', default='data/raw')
    p.add_argument('--out', default='data/prepared/bpe8192')
    p.add_argument('--vocab-size', type=int, default=8192)
    args = p.parse_args()
    if not 260 <= args.vocab_size <= 65535:
        p.error('--vocab-size must be between 260 and 65535')
    raw, out = Path(args.raw).resolve(), Path(args.out).resolve()
    sizes = check_layout(raw)
    out.mkdir(parents=True, exist_ok=True)
    if not (out / 'tokenizer.json').exists() and shutil.disk_usage(out).free < 15 * 1024**3:
        raise RuntimeError('Please leave at least 15 GiB free for preparation')
    paths = train_files(raw)
    identity = [(f.name, sha256_file(f)) for f in paths]
    signature = hashlib.sha256(json.dumps(
        {'revision': REVISION, 'train_files': identity, 'vocab_size': args.vocab_size,
         'min_frequency': 2, 'pretokenizer': 'ByteLevel, no prefix',
         'split': 'sha256-normalized-question-first8-mod100-lt5', 'special': SPECIAL},
        sort_keys=True).encode()).hexdigest()
    tok, tok_info = fit_tokenizer(paths, out / 'tokenizer.json', signature, args.vocab_size)
    if tok_info['counts']['train_rows_used'] + tok_info['counts']['validation_rows_excluded'] != sizes['train']:
        raise ValueError('Tokenizer did not process all official training rows')

    results, offset = [], 0
    print('Encoding ALL rows without truncation...', flush=True)
    for i, path in enumerate(paths, 1):
        dest = out / (path.stem + '.arrow')
        result = encode_shard(path, dest, tok, tok_info['sha256'], row_start=offset)
        results.append(result)
        offset += result['rows']
        print(f'Tokenization [{i}/{len(paths)}]: {offset:,} rows', flush=True)
    if offset != sizes['train']:
        raise ValueError('Encoded training row count does not match raw data')
    test_result = encode_shard(raw / 'data/test/test.parquet', out / 'test.arrow',
                               tok, tok_info['sha256'], test=True)
    if test_result['rows'] != sizes['test']:
        raise ValueError('Encoded test row count does not match raw data')
    report = {'complete': True, 'revision': REVISION, 'tokenizer': tok_info,
              'train_and_validation': summarize(results),
              'test_rows': test_result['rows'], 'sequence_format': 'text token IDs followed by EOS; no BOS or padding stored',
              'cache_format': 'Arrow IPC; uint16 token IDs, int64 original row_id; no truncation',
              'prepared_directory': str(out),
              'versions': {n: importlib.metadata.version(n) for n in ['tokenizers', 'pyarrow']},
              'model_trained': False}
    write_json(out / 'preprocessing.json', report)
    write_json('reports/preprocessing.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    import zipfile
    with zipfile.ZipFile('hw2_preprocessing_report.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for file in [Path('reports/preprocessing.json'), out / 'tokenizer.json', out / 'tokenizer.meta.json']:
            z.write(file, file.name)
    print('DONE: hw2_preprocessing_report.zip', flush=True)


if __name__ == '__main__':
    main()
