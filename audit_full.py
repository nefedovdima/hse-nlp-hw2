import argparse
import ast
import json
import math
import time
from pathlib import Path

from data import REVISION, check_layout, is_validation, iter_rows, train_files, write_json


def quantiles(hist):
    n = sum(hist.values())
    if not n:
        return {}
    values = sorted(hist.items())
    result = {}
    for q in [0, 50, 90, 95, 99, 100]:
        rank = max(1, math.ceil(n * q / 100))
        seen = 0
        for length, count in values:
            seen += count
            if seen >= rank:
                result[str(q)] = length
                break
    return result


def audit(raw, reports, syntax=False, expected_train=None,
          expected_test=None, expected_files=None):
    overrides = {k: v for k, v in [('expected_train', expected_train),
                 ('expected_test', expected_test), ('expected_files', expected_files)]
                 if v is not None}
    counts = check_layout(raw, **overrides)
    reports = Path(reports)
    reports.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    stats = {'revision': REVISION, 'official_counts': counts, 'scanned_rows': 0,
             'usable_train_rows': 0, 'usable_validation_rows': 0,
             'empty_or_nonstring_rows': 0, 'syntax_checked': syntax,
             'syntax_errors': 0 if syntax else None,
             'missing_top_level_solve': 0 if syntax else None}
    hist = {'question': {}, 'code': {}}
    examples = []
    paths = train_files(raw)
    for i, path in enumerate(paths, 1):
        for row in iter_rows([path]):
            stats['scanned_rows'] += 1
            if any(not isinstance(row[k], str) or not row[k].strip()
                   for k in ['question', 'code']):
                stats['empty_or_nonstring_rows'] += 1
                continue
            split = 'validation' if is_validation(row['question']) else 'train'
            stats[f'usable_{split}_rows'] += 1
            for key in hist:
                n = len(row[key])
                hist[key][n] = hist[key].get(n, 0) + 1
            if syntax:
                try:
                    tree = ast.parse(row['code'])
                    if not any(isinstance(x, ast.FunctionDef) and x.name == 'solve'
                               for x in tree.body):
                        stats['missing_top_level_solve'] += 1
                except (SyntaxError, ValueError, RecursionError):
                    stats['syntax_errors'] += 1
            if len(examples) < 3:
                examples.append(row)
        stats['character_lengths'] = {k: quantiles(v) for k, v in hist.items()}
        stats['elapsed_seconds'] = round(time.perf_counter() - start, 2)
        stats['complete'] = False
        stats['files_processed'] = i
        write_json(reports / 'audit_progress.json', stats)
        print(f'[{i}/{len(paths)}] {path.name}: scanned {stats["scanned_rows"]:,}', flush=True)
    if stats['scanned_rows'] != counts['train']:
        raise RuntimeError('Not all train rows were scanned')
    stats['complete'] = True
    stats['scope'] = 'Every row of every local official train shard; no sampling'
    stats['split_rule'] = 'sha256(whitespace-normalized, casefolded question), first 8 bytes big-endian modulo 100 < 5'
    stats['notes'] = [
        'Raw data unchanged. No model training or pass@1 evaluation performed.',
        'Only exact normalized questions are guaranteed to share a split; paraphrases may cross splits.',
        'Length statistics are in characters, not tokens.',
        'No reference code executed; no additional targets created.',
    ]
    write_json(reports / 'audit_full.json', stats)
    write_json(reports / 'examples.json', examples)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
    return stats


def main():
    p = argparse.ArgumentParser(description='Scan every official train row from local disk')
    p.add_argument('--raw', default='data/raw')
    p.add_argument('--reports', default='reports')
    p.add_argument('--syntax', action='store_true', help='Also parse every program with ast; slower')
    args = p.parse_args()
    audit(args.raw, args.reports, args.syntax)


if __name__ == '__main__':
    main()
