"""Small artifact helpers; experiment orchestration stays in its entry point."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def append_log(path, message):
    line = f'{datetime.now(timezone.utc).isoformat(timespec="seconds")} {message}'
    print(line, flush=True)
    with Path(path).open('a') as stream:
        stream.write(line + '\n')
