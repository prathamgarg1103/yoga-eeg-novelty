"""Device-aware loaders for the custom yoga EEG dataset.

Two recording devices are present, mapped onto states:
  * Emotiv EPOC Flex  -> Relaxed / Concentrated   (channels EEG.F7, EEG.F8 @128 Hz)
  * Muse              -> Neutral                  (channels AF7, AF8 @~256 Hz)

Both are normalised into a single ``Recording`` object so downstream feature code
never has to know which device produced the signal.

Auxiliary exports (.edf, .json, *_intervalMarker.csv) are redundant copies of the
primary CSVs and are ignored by ``iter_recording_paths``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from yoga_impact import config

_TS_TOKEN = re.compile(r"(\d{4}\.\d{2}\.\d{2}T\d{2}\.\d{2}\.\d{2})")


@dataclass
class Recording:
    """One physical EEG recording, device-normalised."""
    subject: str
    state: str                      # Relaxed / Concentrated / Neutral
    device: str                     # Emotiv / Muse
    path: Path
    recording_id: str               # grouping key (trials sharing it = one recording)
    fs: float
    channels: list[str]             # e.g. ["F7", "F8"] or ["AF7", "AF8"]
    data: np.ndarray                # shape (n_channels, n_samples), microvolts, float32
    title_internal: str | None = None   # device-reported title (for integrity audit)
    flags: list[str] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return int(self.data.shape[1])

    @property
    def duration_sec(self) -> float:
        return self.n_samples / self.fs if self.fs else float("nan")

    @property
    def label_mismatch(self) -> bool:
        """True if the device-reported title disagrees with the folder label."""
        if not self.title_internal:
            return False
        t = self.title_internal.lower()
        return (self.subject.lower() not in t) or (self.state.lower() not in t)


# ---------------------------------------------------------------------------
# File discovery / classification
# ---------------------------------------------------------------------------
def _is_auxiliary(name: str) -> bool:
    low = name.lower()
    if low.endswith((".json", ".edf")):
        return True
    if low.endswith("_intervalmarker.csv"):
        return True
    return False


def classify_device(path: Path, state: str) -> str:
    name = path.name
    if "EPOCFLEX" in name.upper() or name.lower().endswith((".md.csv", ".md.bp.csv")):
        return "Emotiv"
    # Muse files are the bare "<Subject>-Neutral-<N>.csv" exports.
    return "Muse"


def iter_recording_paths(root: Path | None = None):
    """Yield (subject, state, device, path) for every *primary* EEG file."""
    root = root or config.CUSTOM_ROOT
    for subj_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        subject = subj_dir.name
        for state_dir in sorted(p for p in subj_dir.iterdir() if p.is_dir()):
            state = state_dir.name
            for path in sorted(state_dir.rglob("*.csv")):
                if _is_auxiliary(path.name):
                    continue
                yield subject, state, classify_device(path, state), path


# ---------------------------------------------------------------------------
# Metadata parsing (Emotiv)
# ---------------------------------------------------------------------------
def _parse_emotiv_meta(first_line: str) -> dict:
    meta: dict[str, str] = {}
    for part in first_line.split(","):
        if ":" in part:
            k, _, v = part.partition(":")
            meta[k.strip().lower()] = v.strip()
    return meta


def _recording_id(subject: str, state: str, meta: dict, path: Path) -> str:
    ts = meta.get("start timestamp")
    if ts:
        try:
            return f"{subject}_{state}_{int(float(ts))}"
        except ValueError:
            pass
    m = _TS_TOKEN.search(path.name)
    if m:
        return f"{subject}_{state}_{m.group(1)}"
    return f"{subject}_{state}_{path.stem}"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _load_emotiv(subject: str, state: str, path: Path) -> Recording:
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        first = fh.readline()
    has_meta = first.lower().startswith("title:") or "headset type" in first.lower()
    meta = _parse_emotiv_meta(first) if has_meta else {}
    df = pd.read_csv(path, skiprows=1 if has_meta else 0, low_memory=False)

    cols = {c.upper(): c for c in df.columns}
    f7 = cols.get("EEG.F7")
    f8 = cols.get("EEG.F8")
    if f7 is None or f8 is None:
        raise ValueError(f"EEG.F7/EEG.F8 not found in {path.name}")

    arr = df[[f7, f8]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32).T
    # keep columns (samples) where at least one channel is finite
    keep = np.isfinite(arr).any(axis=0)
    arr = arr[:, keep]

    fs = config.EMOTIV_FS
    sr = meta.get("sampling rate", "")
    m = re.search(r"eeg_(\d+)", sr)
    if m:
        fs = float(m.group(1))

    return Recording(
        subject=subject, state=state, device="Emotiv", path=path,
        recording_id=_recording_id(subject, state, meta, path),
        fs=fs, channels=["F7", "F8"], data=arr,
        title_internal=meta.get("title"),
    )


def _load_muse(subject: str, state: str, path: Path) -> Recording:
    df = pd.read_csv(path, low_memory=False)
    cols = {c.strip().upper(): c for c in df.columns}
    af7 = cols.get("AF7")
    af8 = cols.get("AF8")
    if af7 is None or af8 is None:
        raise ValueError(f"AF7/AF8 not found in {path.name}")

    arr = df[[af7, af8]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32).T
    keep = np.isfinite(arr).any(axis=0)
    arr = arr[:, keep]

    fs = config.MUSE_FS
    ts_col = cols.get("TIMESTAMPS") or cols.get("TIMESTAMP")
    if ts_col is not None:
        ts = pd.to_numeric(df[ts_col], errors="coerce").to_numpy()
        dt = np.diff(ts)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if dt.size:
            fs = float(round(1.0 / np.median(dt)))

    return Recording(
        subject=subject, state=state, device="Muse", path=path,
        recording_id=f"{subject}_{state}_{path.stem}",
        fs=fs, channels=["AF7", "AF8"], data=arr,
    )


def load_recording(subject: str, state: str, device: str, path: Path) -> Recording:
    rec = _load_emotiv(subject, state, path) if device == "Emotiv" \
        else _load_muse(subject, state, path)
    # quality flags
    for i, ch in enumerate(rec.channels):
        x = rec.data[i]
        finite = x[np.isfinite(x)]
        if finite.size == 0:
            rec.flags.append(f"{ch}:empty")
        elif float(np.std(finite)) < 1e-6:
            rec.flags.append(f"{ch}:flat")
    if rec.duration_sec < config.WIN_SEC:
        rec.flags.append("too_short")
    if rec.label_mismatch:
        rec.flags.append(f"title_mismatch:{rec.title_internal}")
    return rec


def load_all(states: tuple[str, ...] | None = None, device: str | None = None):
    """Load every recording, optionally filtered by state and/or device."""
    out: list[Recording] = []
    for subject, state, dev, path in iter_recording_paths():
        if states and state not in states:
            continue
        if device and dev != device:
            continue
        try:
            out.append(load_recording(subject, state, dev, path))
        except Exception as exc:  # noqa: BLE001 - audit should not crash on one bad file
            print(f"[load error] {path.name}: {exc}")
    return out
