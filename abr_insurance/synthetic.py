# synthetic.py — Synthetic lapse experience data generator
#
# Generates a synthetic dataset matching the SOA Post-Level Term cell
# structure for pipeline verification purposes.
#
# Declared measurement mapping for synthetic data:
#   This is NOT SOA experience data.
#   It is a synthetic dataset calibrated to match SOA aggregate lapse
#   rates from the 2014 Post-Level Term study (10-year term):
#
#     Duration 6-9:  ~7%  lapse rate (level term, stable)
#     Duration 10:   ~67% lapse rate (shock lapse at end of level term)
#     Duration 11:   ~35% lapse rate (post-level, elevated)
#     Duration 12:   ~13% lapse rate (post-level, declining)
#     Duration 13+:  ~8%  lapse rate (post-level, stabilizing)
#
#   Cell dimensions are declared to match SOA study design:
#     IssueYear:       2000-2009
#     Duration:        6-14 (individual years, not grouped)
#     PremiumJumpBand: 1-5 (ordinal, representing increasing jump magnitude)
#
#   Within-cell variation is synthetic — drawn from a declared
#   distribution around the SOA aggregate calibration targets.
#   Premium jump effect is declared as: higher jump band → higher
#   lapse rate at duration 10, lower retention in post-level years.
#
# This data is admissible for pipeline verification only.
# Replace with actual SOA or Symetra experience data for production use.
#
# Bounded over D. No claim beyond D.
# Metatron Dynamics, Inc.

import numpy as np
import pandas as pd
from typing import Optional


# ---- SOA calibration targets -------------------------------------------

# Aggregate lapse rates from SOA 2014 study (by duration group)
# Source: SOA Post-Level Term Lapse and Mortality Study, 2014
SOA_CALIBRATION = {
    6:  0.070,   # level term, stable
    7:  0.070,
    8:  0.070,
    9:  0.070,
    10: 0.674,   # shock lapse — end of level term
    11: 0.350,   # first post-level year
    12: 0.131,   # second post-level year
    13: 0.081,   # stabilizing
    14: 0.075,
}

# Premium jump effect: multiplier applied to base rate at each duration
# Higher jump band → higher shock lapse at duration 10
# Declared: ordinal bands 1-5, calibration is synthetic
JUMP_EFFECT = {
    1: 0.70,   # lowest jump — lowest shock lapse
    2: 0.85,
    3: 1.00,   # reference band
    4: 1.15,
    5: 1.30,   # highest jump — highest shock lapse
}

# Issue year effect: mild cohort trend — later cohorts slightly lower lapse
# Declared synthetic — no SOA cohort data available at this aggregation
COHORT_EFFECT = {
    2000: 1.05,
    2001: 1.03,
    2002: 1.02,
    2003: 1.01,
    2004: 1.00,
    2005: 0.99,
    2006: 0.98,
    2007: 0.97,
    2008: 0.96,
    2009: 0.95,
}


# ---- Synthetic data generation -----------------------------------------

def generate_synthetic_lapse_data(
    issue_years: Optional[list] = None,
    durations: Optional[list] = None,
    jump_bands: Optional[list] = None,
    noise_scale: float = 0.02,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic lapse experience data calibrated to SOA aggregates.

    Parameters
    ----------
    issue_years  : list of issue years (default 2000-2009)
    durations    : list of duration years (default 6-14)
    jump_bands   : list of premium jump bands (default 1-5, ordinal)
    noise_scale  : standard deviation of synthetic within-cell noise
    seed         : random seed for reproducibility

    Returns
    -------
    DataFrame with columns:
        IssueYear, Duration, PremiumJumpBand, LapseRate

    Declared constraints:
        - LapseRate clipped to [0.001, 0.999] — domain D
        - Within-cell noise declared as N(0, noise_scale)
        - Premium jump effect declared as multiplicative scalar
        - Cohort effect declared as multiplicative scalar
        - All parameters declared above — no silent assumptions
    """
    rng = np.random.default_rng(seed)

    if issue_years is None:
        issue_years = list(range(2000, 2010))
    if durations is None:
        durations = list(range(6, 15))
    if jump_bands is None:
        jump_bands = [1, 2, 3, 4, 5]

    rows = []
    for year in issue_years:
        for dur in durations:
            for jump in jump_bands:
                # Base rate from SOA calibration
                base = SOA_CALIBRATION.get(dur, 0.075)

                # Apply declared multiplicative effects
                jump_mult = JUMP_EFFECT.get(jump, 1.0)
                cohort_mult = COHORT_EFFECT.get(year, 1.0)

                # Synthetic within-cell noise
                noise = rng.normal(0.0, noise_scale)

                # Composed lapse rate
                rate = base * jump_mult * cohort_mult + noise

                # Clip to declared domain D: lapse rate ∈ [0.001, 0.999]
                rate = float(np.clip(rate, 0.001, 0.999))

                rows.append({
                    'IssueYear':        year,
                    'Duration':         dur,
                    'PremiumJumpBand':  jump,
                    'LapseRate':        rate,
                })

    df = pd.DataFrame(rows)
    return df


def save_synthetic_data(filepath: str, **kwargs) -> pd.DataFrame:
    """
    Generate and save synthetic data to CSV.

    Parameters
    ----------
    filepath : output path (CSV)
    **kwargs : passed to generate_synthetic_lapse_data

    Returns
    -------
    DataFrame of generated data
    """
    df = generate_synthetic_lapse_data(**kwargs)
    df.to_csv(filepath, index=False)
    print(f"Synthetic data saved: {filepath}")
    print(f"Shape: {df.shape}")
    print(f"\nDeclared calibration targets (SOA aggregate):")
    for dur, rate in SOA_CALIBRATION.items():
        mean_synthetic = df[df['Duration'] == dur]['LapseRate'].mean()
        print(f"  Duration {dur:2d}: SOA={rate:.3f}  synthetic_mean={mean_synthetic:.3f}")
    print(f"\nDECLARED: This is synthetic data calibrated to SOA aggregates.")
    print(f"NOT SOA experience data. Not Symetra data.")
    print(f"For pipeline verification only.")
    return df


if __name__ == "__main__":
    import os
    outpath = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "synthetic_lapse_data.csv"
    )
    save_synthetic_data(outpath)
