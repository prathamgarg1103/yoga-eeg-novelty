"""Build the CLEAN, de-duplicated custom yoga dataset.

Per the integrity investigation + user decision:
  * de-duplicate by signal hash (identical F7/F8 data = one physical recording),
  * trust the device-assigned internal TITLE as the ground-truth (subject, state),
  * keep Emotiv Relaxed/Concentrated only (Neutral = Muse = excluded, device confound).

Result: 19 unique recordings with trustworthy labels. ``recording_id`` is the signal
hash, so windows from one physical recording can never leak across CV folds, and LOPO
uses the TRUE (title) subject.

Outputs: yoga_impact/outputs/custom_clean.csv
"""
from __future__ import annotations

import hashlib
import re

import numpy as np
import pandas as pd

from yoga_impact import config
from yoga_impact.io_custom import Recording, iter_recording_paths, load_recording

_TITLE = re.compile(r"([A-Za-z]+)-(Relaxed|Concentrated|Neutral)", re.IGNORECASE)


def _sig_hash(data: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(data, dtype=np.float32)).hexdigest()[:12]


def _label_from_title(title: str | None):
    if not title:
        return None, None
    m = _TITLE.match(title.strip())
    if not m:
        return None, None
    return m.group(1).capitalize(), m.group(2).capitalize()


def load_clean_recordings() -> list[Recording]:
    """De-duplicated Emotiv Relaxed/Concentrated recordings, labelled by title."""
    seen: dict[str, Recording] = {}
    for subject, state, dev, path in iter_recording_paths():
        if dev != "Emotiv" or state not in config.HEADLINE_STATES:
            continue
        try:
            rec = load_recording(subject, state, dev, path)
        except Exception as exc:  # noqa: BLE001
            print(f"[load error] {path.name}: {exc}")
            continue

        t_subj, t_state = _label_from_title(rec.title_internal)
        if t_subj is None or t_state not in config.HEADLINE_STATES:
            # fall back to folder label if the title is unparseable
            t_subj, t_state = subject, state

        h = _sig_hash(rec.data)
        if h in seen:
            continue  # duplicate signal already kept

        # override labels with the trusted title; recording_id = signal hash
        rec.subject = t_subj
        rec.state = t_state
        rec.recording_id = h
        seen[h] = rec

    return list(seen.values())


def build_table(recs: list[Recording]) -> pd.DataFrame:
    rows = [
        dict(
            recording_id=r.recording_id,
            subject=r.subject,
            state=r.state,
            label=int(r.state == config.POSITIVE_CLASS),   # 1 = Relaxed
            fs=r.fs,
            n_samples=r.n_samples,
            duration_sec=round(r.duration_sec, 1),
            source_file=r.path.name,
        )
        for r in recs
    ]
    return pd.DataFrame(rows).sort_values(["subject", "state", "recording_id"])


def main() -> None:
    recs = load_clean_recordings()
    df = build_table(recs)
    out = config.OUTPUT_ROOT / "custom_clean.csv"
    df.to_csv(out, index=False)

    line = "=" * 60
    print(line)
    print("CLEAN CUSTOM YOGA DATASET (de-duplicated, title-labelled)")
    print(line)
    print(f"Unique recordings: {len(df)}")
    print(f"Total EEG minutes: {df['duration_sec'].sum() / 60:.1f}")
    print("\n-- recordings per subject x state --")
    print(pd.crosstab(df["subject"], df["state"], margins=True).to_string())
    print("\n-- duration (s) summary per state --")
    print(df.groupby("state")["duration_sec"].describe()[["count", "mean", "min", "max"]].to_string())
    print(f"\nClean manifest -> {out}")
    print(line)


if __name__ == "__main__":
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    main()
