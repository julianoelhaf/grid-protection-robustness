import fcntl
import json
import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile


def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def publish_json(path, value):
    """Publish once, allowing concurrent writers only when their content agrees."""
    data = canonical(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=".manifest-") as staging:
        staging.write(data)
        staging.flush()
        try:
            os.link(staging.name, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError(f"Immutable manifest mismatch: {path}")


def read_json(path):
    return json.loads(path.read_bytes())


@contextmanager
def claim(root, name, wait=False):
    locks = root / ".locks"
    locks.mkdir(parents=True, exist_ok=True)
    with (locks / name).open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
        except BlockingIOError:
            yield False
        else:
            yield True
