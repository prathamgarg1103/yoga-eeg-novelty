"""Central configuration: paths, frequency bands, windowing, seeds.

Every module imports paths/params from here so nothing is hard-coded elsewhere.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent          # E:\yogic
DATA_ROOT = PROJECT_ROOT / "datasets"
CUSTOM_ROOT = DATA_ROOT / "custom_yoga" / "Custom Dataset"      # extracted zip
WESAD_ROOT = DATA_ROOT / "WESAD"
MEDITATION_ROOT = DATA_ROOT / "ds001787-meditation"

PKG_ROOT = PROJECT_ROOT / "yoga_impact"
OUTPUT_ROOT = PKG_ROOT / "outputs"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# EEG signal processing
# ---------------------------------------------------------------------------
# Canonical frequency bands (Hz). Gamma capped at 45 to stay below the 50 Hz mains.
BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

EMOTIV_FS = 128.0          # Emotiv EPOC Flex raw EEG sampling rate
MUSE_FS = 256.0            # Muse raw EEG sampling rate (estimated from timestamps)
EEG_BANDPASS = (1.0, 45.0)
NOTCH_FREQ = 50.0          # India mains frequency

# Feature windowing for the custom EEG model
WIN_SEC = 8.0
WIN_OVERLAP = 0.5          # 50 % overlap

# ---------------------------------------------------------------------------
# Headline task definition
# ---------------------------------------------------------------------------
# Clean, single-device (Emotiv F7/F8) binary task. Neutral (Muse) is EXCLUDED
# from the headline to avoid the device confound.
HEADLINE_STATES = ("Relaxed", "Concentrated")
POSITIVE_CLASS = "Relaxed"      # Relaxation Index = P(Relaxed) x 100
EMOTIV_CHANNELS = ("F7", "F8")
MUSE_CHANNELS = ("AF7", "AF8")

SEED = 42
