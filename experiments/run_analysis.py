# experiments/run_analysis.py
#
# Entry point for ABR insurance lapse detection analysis.
#
# Usage:
#   python experiments/run_analysis.py --filepath data/SOA_lapse_study.xlsx
#
# What this does:
#   1. Prints full session declaration before any operator acts
#   2. Loads SOA Post-Level Term lapse study data
#   3. Builds declared relational structure over experience cells
#   4. Applies ABRCE operators: A → B → R → E
#   5. Computes relational spread and momentum detection signals
#   6. Compares momentum against SOA shock lapse benchmark
#   7. Plots results and saves to experiments/output/
#
# Open condition — column names:
#   SOA Excel column names must be verified against the actual file.
#   If the script fails on load, check column names and pass them
#   explicitly via --lapse-col, --cohort-col, --duration-col, --jump-col.
#
# Open condition — SOA shock duration index:
#   Pass --soa-shock-index after inspecting the SOA data to identify
#   the first post-level duration with elevated lapse rates.
#   Without this, lead time is not computed.
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


# ---- Argument parsing --------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="ABR Insurance Lapse Detection — SOA Post-Level Term Study"
    )
    p.add_argument(
        "--filepath", required=True,
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
        "--soa-shock-index", type=int, default=None,
        help=(
            "Duration index of SOA-reported shock lapse point. "
            "Declare after inspecting SOA data. "
            "Required for lead time computation."
        )
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
        "--signal-threshold", type=float, default=0.10,
        help="Momentum detection threshold (default 0.10)"
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

def plot_results(durations, spread_series, momentum, trigger_result,
                 soa_shock_index, output_dir, momentum_window):
    """
    Two-panel plot:
    Top:    Relational spread by duration
    Bottom: Relational momentum with detection threshold

    Declared projection discards:
    - spread: discards within-duration cell arrangement
    - momentum: discards magnitude beyond unit scale
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    t = np.arange(len(durations))

    # Top panel — relational spread
    ax1.plot(t, spread_series, color='steelblue', lw=1.5,
             label='Relational spread (population variance by construction)')
    if soa_shock_index is not None:
        ax1.axvline(soa_shock_index, color='crimson', ls='--', alpha=0.7,
                    label=f'SOA shock lapse (declared, index={soa_shock_index})')
    if trigger_result.signal_trigger_index is not None:
        ax1.axvline(trigger_result.signal_trigger_index,
                    color='forestgreen', ls=':', lw=2,
                    label=f'Momentum trigger (index={trigger_result.signal_trigger_index})')
    ax1.set_ylabel('Relational Spread')
    ax1.set_title(
        'ABR Insurance Lapse Detection — SOA Post-Level Term Experience\n'
        'Metatron Dynamics, Inc. — Bounded over D. No claim beyond D.'
    )
    ax1.legend(fontsize=8)

    # Bottom panel — relational momentum
    ax2.plot(t, momentum, color='darkorange', lw=1.5,
             label=f'Relational momentum — mean(|C(R(A(W)))|) w={momentum_window}')
    ax2.axhline(trigger_result.signal_threshold,
                color='forestgreen', ls='--', alpha=0.5,
                label=f'Detection threshold ({trigger_result.signal_threshold})')
    if trigger_result.signal_trigger_index is not None:
        ax2.axvline(trigger_result.signal_trigger_index,
                    color='forestgreen', ls=':', lw=2)
    if soa_shock_index is not None:
        ax2.axvline(soa_shock_index, color='crimson', ls='--', alpha=0.7)

    ax2.set_ylabel('Relational Momentum')
    ax2.set_xlabel('Duration Index')
    ax2.legend(fontsize=8)

    # X tick labels — actual duration values
    step = max(1, len(durations) // 10)
    ax2.set_xticks(t[::step])
    ax2.set_xticklabels([str(d) for d in durations[::step]], fontsize=7)

    plt.tight_layout()
    outpath = os.path.join(output_dir, 'abr_lapse_detection.png')
    plt.savefig(outpath, dpi=120)
    plt.show()
    print(f"\nPlot saved → {outpath}")


# ---- Main --------------------------------------------------------------

def main():
    args = parse_args()

    print("=" * 60)
    print("ABR Insurance Lapse Detection")
    print("Metatron Dynamics, Inc.")
    print("Bounded over D. No claim beyond D.")
    print("=" * 60)
    print()

    # ---- Run detection pipeline ----------------------------------------
    result, durations = run_detection(
        filepath=args.filepath,
        sheet_name=args.sheet_name,
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
    print("METRICS")
    print("=" * 60)

    trigger = run_metrics(
        spread_by_duration=result.spread_by_duration,
        momentum=result.momentum,
        durations=list(durations),
        level_term_period=args.level_term,
        signal_threshold=args.signal_threshold,
        soa_shock_duration_index=args.soa_shock_index,
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
        trigger_result=trigger,
        soa_shock_index=args.soa_shock_index,
        output_dir=args.output_dir,
        momentum_window=args.momentum_window,
    )

    # ---- Final declared summary ----------------------------------------
    print("\n" + "=" * 60)
    print("DECLARED RESULT")
    print("=" * 60)
    if trigger.lead_time_steps is not None:
        direction = (
            "before" if trigger.lead_time_steps > 0 else
            "after"  if trigger.lead_time_steps < 0 else "at"
        )
        print(
            f"Relational momentum fires {abs(trigger.lead_time_steps)} "
            f"duration steps {direction} declared SOA shock lapse point."
        )
        print("Lead time is provisional — verify SOA shock index declaration.")
    else:
        print("Lead time not computed.")
        print("Declare --soa-shock-index after inspecting SOA data.")

    print("\nOpen conditions carried forward:")
    for oc in result.open_conditions:
        print(f"  [{oc.status.value.upper()}] {oc.name}")

    print("\nBounded over D. No claim beyond D.")
    print("Metatron Dynamics, Inc.")


if __name__ == "__main__":
    main()
