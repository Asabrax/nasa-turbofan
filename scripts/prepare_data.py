"""Import the twelve checksum-verified C-MAPSS inputs from a local ZIP."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'data/source.json').read_text())['files']
    wanted = [f'{prefix}_FD{i:03d}.txt' for prefix in ('train', 'test', 'RUL') for i in range(1, 5)]
    verified = {}
    with zipfile.ZipFile(args.archive) as archive:
        for name in wanted:
            matches = [item for item in archive.infolist() if Path(item.filename).name == name and not item.is_dir()]
            if len(matches) != 1:
                raise ValueError(f'Expected exactly one {name}; found {len(matches)}')
            content = archive.read(matches[0])
            expected = manifest[f'data/raw/{name}']['sha256']
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError(f'Checksum mismatch: {name}; no inputs written')
            verified[name] = content
    destination = ROOT / 'data/raw'
    destination.mkdir(parents=True, exist_ok=True)
    for name, content in verified.items():
        (destination / name).write_bytes(content)
    print(f'Imported {len(verified)} verified inputs into {destination}')

if __name__ == '__main__':
    main()
