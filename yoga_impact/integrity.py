"""Signal-level integrity probe for the custom Emotiv yoga data.

The audit found internal titles that disagree with folder labels and many shared
start-timestamps. This module hashes the actual F7/F8 signal of every Emotiv
Relaxed/Concentrated recording to discover *true* duplicates -- especially copies
that cross subject or state boundaries (which would be contamination + mislabeling,
and fatal to any honest classifier).

Outputs: yoga_impact/outputs/integrity.csv + printed report.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np
import pandas as pd

from yoga_impact import config
from yoga_impact.io_custom import iter_recording_paths, load_recording


def _sig_hash(data: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(data, dtype=np.float32)).hexdigest()[:12]


def probe() -> pd.DataFrame:
    rows = []
    for subj, state, dev, path in iter_recording_paths():
        if dev != "Emotiv" or state not in config.HEADLINE_STATES:
            continue
        try:
            r = load_recording(subj, state, dev, path)
        except Exception as exc:  # noqa: BLE001
            print(f"[err] {path.name}: {exc}")
            continue
        start_ts = r.recording_id.rsplit("_", 1)[-1]
        rows.append(dict(
            subject=subj, state=state, start_ts=start_ts,
            title_internal=r.title_internal or "", n_samples=r.n_samples,
            sig_hash=_sig_hash(r.data), file=path.name,
        ))
    return pd.DataFrame(rows)


def report(df: pd.DataFrame) -> None:
    line = "=" * 70
    print(line)
    print("SIGNAL-LEVEL INTEGRITY PROBE (Emotiv Relaxed/Concentrated)")
    print(line)
    print(f"Files: {len(df)}")
    print(f"Distinct signal hashes (truly unique recordings): {df['sig_hash'].nunique()}")

    by_hash = defaultdict(list)
    for rec in df.to_dict("records"):
        by_hash[rec["sig_hash"]].append(rec)

    cross_subject = cross_state = within_dups = 0
    print("\n-- duplicate signal groups (identical F7/F8 data) --")
    for h, grp in sorted(by_hash.items(), key=lambda kv: -len(kv[1])):
        if len(grp) == 1:
            continue
        subs = {g["subject"] for g in grp}
        sts = {g["state"] for g in grp}
        tags = []
        if len(subs) > 1:
            tags.append("CROSS-SUBJECT")
            cross_subject += 1
        if len(sts) > 1:
            tags.append("CROSS-STATE")
            cross_state += 1
        if not tags:
            within_dups += 1
        tag = (" [" + ", ".join(tags) + "]") if tags else " [within subject/state]"
        print(f"\nhash {h}  x{len(grp)}  n_samples={grp[0]['n_samples']}{tag}")
        for g in grp:
            print(f"    {g['subject']:8s} {g['state']:13s} title={g['title_internal']:24s} {g['file']}")

    print("\n" + "-" * 70)
    print(f"Signal-duplicate groups crossing SUBJECT: {cross_subject}")
    print(f"Signal-duplicate groups crossing STATE:   {cross_state}")
    print(f"Within subject/state duplicate groups:    {within_dups}")
    print(line)


def main() -> None:
    df = probe()
    out = config.OUTPUT_ROOT / "integrity.csv"
    df.to_csv(out, index=False)
    report(df)
    print(f"\nIntegrity table -> {out}")


if __name__ == "__main__":
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    main()
