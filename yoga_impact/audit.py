"""Data audit: build a manifest of every custom-yoga recording and flag problems.

This runs BEFORE any modelling and is what makes the rebuild trustworthy. It:
  * inventories every primary recording (subject / state / device / fs / duration),
  * derives recording_id so trials that share one physical recording are grouped,
  * flags label-integrity problems (device title vs folder), flat/empty channels,
  * reports the device<->state confound explicitly.

Outputs:  yoga_impact/outputs/manifest.csv  + a printed summary.
"""
from __future__ import annotations

import pandas as pd

from yoga_impact import config
from yoga_impact.io_custom import load_all


def build_manifest() -> pd.DataFrame:
    recs = load_all()
    rows = []
    for r in recs:
        rows.append(
            dict(
                subject=r.subject,
                state=r.state,
                device=r.device,
                fs=r.fs,
                n_channels=len(r.channels),
                channels="+".join(r.channels),
                n_samples=r.n_samples,
                duration_sec=round(r.duration_sec, 1),
                recording_id=r.recording_id,
                title_internal=r.title_internal or "",
                label_mismatch=r.label_mismatch,
                flags=";".join(r.flags),
                file=r.path.name,
            )
        )
    return pd.DataFrame(rows).sort_values(["subject", "state", "recording_id", "file"])


def summarise(df: pd.DataFrame) -> None:
    line = "=" * 70
    print(line)
    print("CUSTOM YOGA DATASET — AUDIT SUMMARY")
    print(line)
    print(f"Total primary recordings (files): {len(df)}")
    print(f"Distinct physical recordings:     {df['recording_id'].nunique()}")
    print(f"Subjects:                         {sorted(df['subject'].unique())}")

    print("\n-- files per subject x state x device --")
    g = (df.groupby(["subject", "state", "device"]).size()
           .rename("files").reset_index())
    print(g.to_string(index=False))

    print("\n-- DISTINCT RECORDINGS per subject x state (leakage check) --")
    gr = (df.groupby(["subject", "state"])["recording_id"].nunique()
            .rename("recordings").reset_index())
    gf = (df.groupby(["subject", "state"]).size().rename("files").reset_index())
    merged = gf.merge(gr, on=["subject", "state"])
    merged["trials_per_recording"] = (merged["files"] / merged["recordings"]).round(1)
    print(merged.to_string(index=False))

    print("\n-- device <-> state confound --")
    print(pd.crosstab(df["state"], df["device"]).to_string())

    print("\n-- data-quality flags --")
    n_mismatch = int(df["label_mismatch"].sum())
    print(f"Label/title mismatches: {n_mismatch}")
    if n_mismatch:
        cols = ["subject", "state", "title_internal", "file"]
        print(df.loc[df["label_mismatch"], cols].to_string(index=False))
    flagged = df[df["flags"].str.contains("flat|empty|too_short", na=False)]
    print(f"Flat/empty/too-short recordings: {len(flagged)}")
    if len(flagged):
        print(flagged[["subject", "state", "duration_sec", "flags", "file"]].to_string(index=False))

    print("\n-- HEADLINE-TASK pool (Emotiv, Relaxed vs Concentrated) --")
    head = df[(df["device"] == "Emotiv") & (df["state"].isin(config.HEADLINE_STATES))]
    hp = (head.groupby(["subject", "state"])["recording_id"].nunique()
            .rename("recordings").reset_index())
    print(hp.to_string(index=False))
    print(f"\nHeadline files: {len(head)} | distinct recordings: {head['recording_id'].nunique()}")
    print(line)


def main() -> None:
    df = build_manifest()
    out = config.OUTPUT_ROOT / "manifest.csv"
    df.to_csv(out, index=False)
    summarise(df)
    print(f"\nManifest written -> {out}")


if __name__ == "__main__":
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    main()
