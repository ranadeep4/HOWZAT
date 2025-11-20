"""Recompress joblib files in-place or to a new path.

Usage:
    python tools/recompress_joblib.py <input_path> [--out <output_path>] [--method lzma] [--level 9]

Examples:
    # recompress in-place using lzma level 9
    python tools/recompress_joblib.py artifacts\RandomForest_tuned.joblib --method lzma --level 9

This script will load the object with joblib, then dump it again with the requested compression.
Be mindful of memory usage when loading large models.
"""
import argparse
from pathlib import Path
import joblib
import sys

METHOD_MAP = {
    'lzma': ('lzma',),
    'zlib': ('zlib',),
    'gzip': ('gzip',),
    'none': None,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('input', help='Input .joblib file path')
    p.add_argument('--out', '-o', help='Output path. If omitted, overwrites input', default=None)
    p.add_argument('--method', choices=['lzma','zlib','gzip','none'], default='lzma')
    p.add_argument('--level', type=int, default=9, help='Compression level (1-9) where applicable')
    return p.parse_args()


def main():
    args = parse_args()
    inp = Path(args.input)
    if not inp.exists():
        print(f'Input file not found: {inp}', file=sys.stderr)
        sys.exit(2)

    out = Path(args.out) if args.out else inp

    print(f'Loading {inp} ...')
    obj = joblib.load(inp)

    if args.method == 'none':
        compress = 0
    else:
        compress = (args.method, args.level)

    print(f'Saving to {out} with compression={compress} ...')
    joblib.dump(obj, out, compress=compress)
    print('Done.')


if __name__ == '__main__':
    main()
