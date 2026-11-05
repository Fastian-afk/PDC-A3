#!/usr/bin/env python3
"""Imaad Fazal | 23i-0656 | Section C | Assignment 3.
Package source handoff only; exclude binaries, Git metadata and measurements.
"""
import argparse
import hashlib
from pathlib import Path
import stat
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FOLDER = '23i-0656-C'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'dist' / 'PDC-A3-cluster-handoff.zip')
    args = parser.parse_args()
    files = [ROOT / name for name in ('Makefile', 'run.sh', 'README.md', '.gitignore')]
    for directory, suffixes in [('Q1', {'.c', '.h'}), ('Q2', {'.c', '.h'}),
                                ('Q3', {'.c', '.h'}), ('common', {'.h'}),
                                ('tools', {'.py', '.md'}), ('docs', {'.md'})]:
        files += [p for p in (ROOT / directory).rglob('*')
                  if p.is_file() and p.suffix in suffixes and '__pycache__' not in p.parts]
    files = sorted(files, key=lambda p: p.relative_to(ROOT).as_posix())
    if len([p for p in files if p.suffix == '.c']) != 13:
        raise SystemExit('expected13 C entry points; refusing incomplete package')
    for p in files:
        if p.is_symlink() or not p.is_file():
            raise SystemExit(f'expected regular source file: {p}')
    manifest = ''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT).as_posix()}\n'
                       for p in files)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in files:
            z.write(p, f'{FOLDER}/{p.relative_to(ROOT).as_posix()}')
        z.writestr(f'{FOLDER}/MANIFEST.sha256', manifest)
    with tempfile.TemporaryDirectory(prefix='pdc-package-check-') as temp:
        with zipfile.ZipFile(args.output) as z:
            bad = z.testzip()
            if bad:
                raise SystemExit(f'corrupt archive member: {bad}')
            z.extractall(temp)
        extracted = Path(temp) / FOLDER
        for p in files:
            rel = p.relative_to(ROOT)
            if hashlib.sha256((extracted / rel).read_bytes()).digest() != hashlib.sha256(p.read_bytes()).digest():
                raise SystemExit(f'extraction checksum mismatch: {rel}')
        # Zip metadata preserves executable flags; Python extraction itself
        # does not restore permissions on every platform. README offers bash too.
        with zipfile.ZipFile(args.output) as z:
            if not (z.getinfo(f'{FOLDER}/run.sh').external_attr >> 16) & stat.S_IXUSR:
                raise SystemExit('run.sh executable flag missing')
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(args.output.suffix + '.sha256').write_text(f'{digest}  {args.output.name}\n')
    print(f'{args.output}: {len(files) + 1} files; ZIP integrity and extracted SHA-256 checks PASS')
    print(f'SHA-256 {digest}')


if __name__ == '__main__':
    main()
