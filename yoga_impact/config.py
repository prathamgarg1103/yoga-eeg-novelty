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

# ---------------------------------------------------------------------------
# Novelty layer (C1-C5) — all leak-free, all reuse modeling.group_cv_evaluate
# ---------------------------------------------------------------------------
# C1: connectivity / coupling / aperiodic EEG features ----------------------
USE_CONNECTIVITY_FEATURES = True
CONN_BANDS = ("theta", "alpha", "beta")          # bands for coherence + PLV
PAC_PHASE_BAND = "theta"                          # phase-providing band for PAC
PAC_AMP_BANDS = ("beta", "gamma")                 # amplitude-providing bands
PAC_N_BINS = 18                                   # phase bins for Tort MI
APERIODIC_FIT_RANGE = (2.0, 40.0)                 # Hz range for 1/f log-log fit
# Amplitude/offset terms are device-gain dependent: keep them off the cross-device
# transfer feature set (ds001787 <- custom). aperiodic_exponent (slope) IS invariant.
TRANSFER_INVARIANT_ONLY = True
GAIN_DEPENDENT_FEATURES = ("aperiodic_offset",)   # excluded when invariant-only

# C2: 64-channel brain-network analysis (ds001787) --------------------------
NETWORK_MONTAGE = "biosemi64"
NETWORK_CHANNELS = None                           # None = all 64 scalp channels
NETWORK_METHOD = "wpli"                           # wpli | plv | coherence
NETWORK_BANDS = ("theta", "alpha", "beta")
NETWORK_CROP_SEC = 300.0                          # crop each recording for compute parity
NETWORK_WIN_SEC = 4.0                             # connectivity window length
NETWORK_DENSITY = 0.2                             # proportional graph threshold

# C3: uncertainty + convergent validity -------------------------------------
CONFORMAL_ALPHA = 0.1                             # 1-alpha = nominal coverage (90%)
ECE_BINS = 10

# C4: leak-free personalization ---------------------------------------------
# WESAD autonomic discrimination saturates for most condition pairs; baseline-vs-amusement
# is the pair with real headroom (AUROC ~0.73), so it is the informative personalization
# testbed. The per-subject calibration anchor is a DISJOINT condition (meditation), never a
# scored or trained class — so it cannot leak.
PERSON_TASK_LABELS = (1, 3)                       # baseline vs amusement (headroom)
PERSON_POS_LABEL = 3                              # positive class = amusement
PERSON_REF_LABEL = 4                              # meditation windows = adaptation anchor
PERSON_ADAPT_FRACTION = 0.5                       # fraction of anchor windows used to adapt
PERSON_MIN_ADAPT_WINDOWS = 3
PERSON_METHODS = ("recenter", "zscore")

# C5: deep vs classical head-to-head ----------------------------------------
DEEP_BACKEND = "auto"                             # auto -> torch if available else mlp
DEEP_INPUT = "feature_seq"                        # feature_seq | raw_window
DEEP_EPOCHS = 60
DEEP_HIDDEN = 32
DEEP_MAX_SEQ = 64                                 # cap windows per group sequence

# ---------------------------------------------------------------------------
# FLAGSHIP — "Geometry of Calm": Riemannian manifold + optimal-transport (R1-R3)
# ---------------------------------------------------------------------------
# Each window -> an augmented multiband covariance (an SPD matrix). The two frontal
# channels are expanded into band-filtered virtual channels, so a single covariance
# carries spectral power (diagonal), spatial coupling and cross-band coupling
# (off-diagonal). Tangent-space projection at the per-fold geometric mean is the
# feature map; everything stays inside modeling's leak-free group CV.
RIEMANN_BANDS = ("theta", "alpha", "beta")        # virtual-channel bands (per EEG channel)
RIEMANN_SHRINKAGE = 0.10                           # covariance shrinkage toward scaled I
RIEMANN_CALIBRATE = "sigmoid"                      # in-fold probability calibration
# R2 transfer: Emotiv (custom) -> BioSemi (ds001787) frontal manifold, with vs without
# Riemannian re-centering (each domain whitened by its own geometric mean = the
# device-invariance step). Reported honestly either way.
RIEMANN_TRANSFER = True
# R3 optimal transport: Bures-Wasserstein 2-Wasserstein distance between the per-state
# physiological distributions = a single interpretable "how far yoga moves you" effect
# size, with a group-level permutation p-value.
OT_N_PERM = 500
OT_GAUSS_SHRINK = 0.05
# WESAD condition pairs scored for the OT yoga-impact effect size (label ints from wesad.py)
OT_WESAD_PAIRS = (("meditation", "stress"), ("meditation", "baseline"), ("baseline", "stress"))
