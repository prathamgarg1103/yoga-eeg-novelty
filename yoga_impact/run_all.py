"""End-to-end pipeline runner for the yoga-impact project.

Usage:
    python -m yoga_impact.run_all                # full pipeline
    python -m yoga_impact.run_all --stages audit clean wesad custom ds report

Stages (each is independently runnable and caches expensive work):
    audit       - inventory + signal-integrity probe of the custom data
    clean       - build the de-duplicated, title-labelled clean custom set
    wesad       - WESAD autonomic axis (LOSO)              [backbone]
    custom      - custom yoga EEG relaxation model + C1 connectivity ablation [case study]
    ds          - ds001787 external EEG validation (F7/F8 transfer)
    network     - C2: 64-channel brain-network analysis (ds001787)
    personalize - C4: leak-free few-shot subject adaptation
    deep        - C5: deep vs calibrated-classical head-to-head
    uncertainty - C3: conformal intervals + ECE + cross-modal convergent validity
    riemann     - FLAGSHIP: Riemannian manifold + optimal-transport effect size
    report      - assemble the Composite Yoga Index report
"""
from __future__ import annotations

import argparse

ALL_STAGES = ["audit", "clean", "wesad", "custom", "ds",
              "network", "personalize", "deep", "uncertainty", "riemann", "report"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the yoga-impact pipeline")
    ap.add_argument("--stages", nargs="*", default=ALL_STAGES,
                    choices=ALL_STAGES, help="subset of stages to run, in order")
    args = ap.parse_args()

    for stage in args.stages:
        print(f"\n########## STAGE: {stage} ##########")
        if stage == "audit":
            from yoga_impact import audit, integrity
            audit.main()
            integrity.main()
        elif stage == "clean":
            from yoga_impact import clean_custom
            clean_custom.main()
        elif stage == "wesad":
            from yoga_impact import wesad_model
            wesad_model.run()
        elif stage == "custom":
            from yoga_impact import eeg_model_custom
            eeg_model_custom.run()
        elif stage == "ds":
            from yoga_impact import ds_meditation
            ds_meditation.run()
        elif stage == "network":
            from yoga_impact import ds_network
            ds_network.run()
        elif stage == "personalize":
            from yoga_impact import personalization
            personalization.run()
        elif stage == "deep":
            from yoga_impact import deep_baseline
            deep_baseline.run()
        elif stage == "uncertainty":
            from yoga_impact import uncertainty
            uncertainty.run()
        elif stage == "riemann":
            from yoga_impact import riemann
            riemann.run()
        elif stage == "report":
            from yoga_impact import composite
            composite.main()

    print("\nDONE. Artifacts in yoga_impact/outputs/")


if __name__ == "__main__":
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    main()
