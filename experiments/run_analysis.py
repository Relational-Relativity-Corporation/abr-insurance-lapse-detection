# experiments/run_analysis.py
#
# Entry point for ABR insurance lapse relational spread analysis.
#
# Usage:
#   python experiments/run_analysis.py --synthetic
#   python experiments/run_analysis.py --filepath data/SOA_lapse_study.xlsx
#
# What this does:
#   1. Prints full session declaration before any operator acts
#   2. Loads SOA Post-Level Term lapse study data
#   3. Builds declared relational structure over experience cells
#   4. Applies ABRCE operators: A → B → R → E
#   5. Computes relational spread by duration (primary signal)
#   6. Computes relational momentum over spread series (secondary)
#   7. Characterizes spread profile: peak location and elevation ratio
#   8. Plots results and saves to experiments/output/
#
# Primary finding:
#   Relational spread across the declared lapse rate field, observed
#   by duration step. Spread elevation is the finding — not thresholded.
#   Peak location and elevation ratio are declared outputs.
#
# Open condition — column names:
#   SOA Excel column names must be verified against the actual file.
#   If the script fails on load, check column names and pass them
#   explicitly via --lapse-col, --cohort-col, --duration-col, --jump-col.
#
# Bounded over D. No claim beyond D.
# Metatron Dynamics, Inc.

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abr_insurance.detection import run_detection
from abr_insurance.metrics import run_metrics
from abr_insurance.synthetic import save_synthetic_data


# ---- Argument parsing --------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "ABR Insurance Lapse — Relational Spread Analysis\n"
            "Metatron Dynamics, Inc. — Bounded over D. No claim beyond D."
        )
    )
    p.add_argument(
        "--synthetic", action="store_true",
        help=(
            "Use synthetic data calibrated to SOA aggregates. "
            "Declared: NOT SOA or Symetra data. Pipeline verification only."
        )
    )
    p.add_argument(
        "--filepath", default=None,
        help="Path to SOA Post-Level Term lapse study Excel file"
    )
    p.add_argument(
        "--sheet-name", default="Lapse Study",
        help="Excel sheet name (open condition — verify against actual file)"
    )
    p.add_argument(
        "--lapse-col", default="LapseRate",
        help="Lapse rate column name (open condition — verify)"
    )
    p.add_argument(
        "--cohort-col", default="IssueYear",
        help="Issue year column name (open condition — verify)"
    )
    p.add_argument(
        "--duration-col", default="Duration",
        help="Duration column name (open condition — verify)"
    )
    p.add_argument(
        "--jump-col", default="PremiumJumpBand",
        help="Premium jump band column name (open condition — verify)"
    )
    p.add_argument(
        "--level-term", type=int, default=10,
        help="Level term period in years (default 10)"
    )
    p.add_argument(
        "--rho-base", type=float, default=0.4,
        help="Local contrast scaling (default 0.4)"
    )
    p.add_argument(
        "--momentum-window", type=int, default=5,
        help="Rolling window for momentum signal (default 5)"
    )
    p.add_argument(
        "--no-jump", action="store_true",
        help="Exclude premium jump adjacency from declared relations"
    )
    p.add_argument(
        "--output-dir", default="experiments/output",
        help="Directory for plot output"
    )
    return p.parse_args()


# ---- Plotting ----------------------------------------------------------

def plot_results(durations, spread_series, momentum, profile, mom_char,
                 output_dir, momentum_window):
    """
    Two-panel plot:
    Top:    Relational spread by duration (primary signal)
    Bottom: Relational momentum over spread series (secondary)

    Declared projection discards:
    - spread: discards within-duration cell arrangement
    - momentum: discards magnitude beyond unit scale
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    t = np.arange(len(durations))

    # Top panel — relational spread (primary)
    ax1.plot(t, spread_series, color='steelblue', lw=1.5,
             label='Relational spread in lapse experience field (primary signal)')

    if profile.peak_duration_index is not None:
        ax1.axvline(profile.peak_duration_index,
                    color='steelblue', ls='--', alpha=0.6,
                    label=f'Spread peak (index={profile.peak_duration_index}, '
                          f'duration={durations[profile.peak_duration_index]})')
        if profile.baseline_mean is not None:
            ax1.axhline(profile.baseline_mean,
                        color='gray', ls=':', lw=1.0, alpha=0.7,
                        label=f'Pre-peak baseline mean ({profile.baseline_mean:.4f})')
        if profile.elevation_ratio is not None:
            ax1.annotate(
                f'{profile.elevation_ratio:.1f}x elevation',
                xy=(profile.peak_duration_index,
                    float(np.nanmax(spread_series))),
                xytext=(profile.peak_duration_index + 0.5,
                        float(np.nanmax(spread_series)) * 0.92),
                fontsize=8, color='steelblue',
            )

    ax1.set_ylabel('Relational Spread')
    ax1.set_title(
        'ABR Insurance Lapse — Relational Spread Analysis\n'
        'Metatron Dynamics, Inc. — Bounded over D. No claim beyond D.'
    )
    ax1.legend(fontsize=8)

    # Bottom panel — relational momentum (secondary)
    ax2.plot(t, momentum, color='darkorange', lw=1.5,
             label=f'Relational momentum — directional trend over spread series  '
                   f'w={momentum_window}  (secondary characterization)')

    if mom_char.peak_momentum_index is not None:
        ax2.axvline(mom_char.peak_momentum_index,
                    color='darkorange', ls='--', alpha=0.6,
                    label=f'Momentum peak (index={mom_char.peak_momentum_index})')

    ax2.set_ylabel('Relational Momentum')
    ax2.set_xlabel('Duration Index')
    ax2.legend(fontsize=8)
    ax2.text(
        0.01, 0.05,
        'Secondary characterization — does not gate primary finding',
        transform=ax2.transAxes, fontsize=7, color='gray', style='italic'
    )

    # X tick labels — actual duration values
    step = max(1, len(durations) // 10)
    ax2.set_xticks(t[::step])
    ax2.set_xticklabels([str(d) for d in durations[::step]], fontsize=7)

    plt.tight_layout()
    outpath = os.path.join(output_dir, 'abr_lapse_spread.png')
    plt.savefig(outpath, dpi=120)
    plt.show()
    print(f"\nPlot saved → {outpath}")


# ---- Main --------------------------------------------------------------

def main():
    args = parse_args()

    print("=" * 60)
    print("ABR Insurance Lapse — Relational Spread Analysis")
    print("Metatron Dynamics, Inc.")
    print("Bounded over D. No claim beyond D.")
    print("=" * 60)
    print()

    # ---- Synthetic data path -------------------------------------------
    if args.synthetic:
        print("DECLARED: Using synthetic data calibrated to SOA aggregates.")
        print("NOT SOA experience data. NOT Symetra data.")
        print("Pipeline verification only.\n")
        synthetic_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "synthetic_lapse_data.csv"
        )
        save_synthetic_data(synthetic_path)
        filepath = synthetic_path
        sheet_name = None   # CSV — no sheet
    elif args.filepath:
        filepath = args.filepath
        sheet_name = args.sheet_name
    else:
        print("ERROR: provide --filepath or --synthetic")
        sys.exit(1)

    # ---- Run detection pipeline ----------------------------------------
    result, durations = run_detection(
        filepath=filepath,
        sheet_name=sheet_name,
        lapse_col=args.lapse_col,
        cohort_col=args.cohort_col,
        duration_col=args.duration_col,
        jump_col=args.jump_col,
        rho_base=args.rho_base,
        momentum_window=args.momentum_window,
        include_jump=not args.no_jump,
        verbose=True,
    )

    durations = np.array(durations)

    # ---- Run metrics ---------------------------------------------------
    print("\n" + "=" * 60)
    print("METRICS — RELATIONAL SPREAD CHARACTERIZATION")
    print("=" * 60)

    profile, mom_char = run_metrics(
        spread_by_duration=result.spread_by_duration,
        momentum=result.momentum,
        durations=list(durations),
        level_term_period=args.level_term,
        verbose=True,
    )

    # ---- Plot ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("PLOT")
    print("=" * 60)

    plot_results(
        durations=durations,
        spread_series=result.spread_by_duration,
        momentum=result.momentum,
        profile=profile,
        mom_char=mom_char,
        output_dir=args.output_dir,
        momentum_window=args.momentum_window,
    )

    # ---- Final declared summary ----------------------------------------
    print("\n" + "=" * 60)
    print("DECLARED RESULT")
    print("=" * 60)

    if profile.peak_duration_index is not None:
        dur_label = durations[profile.peak_duration_index]
        print(
            f"Relational spread peaks at duration index "
            f"{profile.peak_duration_index} (duration {dur_label})."
        )
        if profile.elevation_ratio is not None:
            print(
                f"Elevation: {profile.elevation_ratio:.1f}x above "
                f"pre-peak baseline mean ({profile.baseline_mean:.6f})."
            )
        print(
            "\nSpread elevation is the declared finding. "
            "No threshold applied. No lead time claimed."
        )
    else:
        print("Spread peak not determined — insufficient data.")
        print("Verify data load and column name declarations.")

    print("\nOpen conditions carried forward:")
    for oc in result.open_conditions:
        print(f"  [{oc.status.value.upper()}] {oc.name}")

    print()
    for oc in profile.open_conditions:
        print(f"  [OPEN] {oc}")

    print("\nBounded over D. No claim beyond D.")
    print("Metatron Dynamics, Inc.")


if __name__ == "__main__":
    main()
