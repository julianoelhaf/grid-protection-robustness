import hashlib
import io
from pathlib import Path

import numpy as np
import pandas as pd


def digest(path, name="sha256"):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, name).hexdigest()


def read_episode(payload, expected_hash, episode_id, channels, rate):
    if hashlib.sha256(payload).hexdigest() != expected_hash:
        raise ValueError(f"Waveform checksum mismatch in episode {episode_id}")
    episode = pd.read_pickle(io.BytesIO(payload)).rename(columns={"time_s": "timestamp"})
    names = [column for column in episode.columns if column != "timestamp"]
    if (channels is not None and names != channels) or len(episode) != rate:
        raise ValueError(f"Channel layout mismatch in episode {episode_id}")
    expected_time = np.arange(1, rate + 1) / rate
    if not np.allclose(episode.timestamp, expected_time, rtol=0, atol=5.1e-7):
        raise ValueError(f"Timestamp grid mismatch in episode {episode_id}")
    return episode, names
