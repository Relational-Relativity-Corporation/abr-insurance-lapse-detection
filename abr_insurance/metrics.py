# metrics.py — Relational spread characterization
#
# Characterizes the relational spread field across the declared duration
# domain. Spread elevation is the primary finding.
#
# What this file measures:
#   - Relational spread by duration: observed scalar variance per step
#   - Peak location: duration index and value of maximum spread
#   - Elevation ratio: peak spread vs pre-peak baseline mean
#   - Spread profile: full series with declared provenance
#
# What this file does not claim:
#   - No threshold-as-detection. Spread is observed, not gated.
#   - No lead time. Duration steps are not calendar time.
#   - No universality. Result bounded to this dataset and declaration.
#   - No causality. Spread correlation with lapse behavior, not causation.
#
# The relational momentum series is reported as secondary characterization
# of directional trend over the spread field — it does not gate findings.
#
# Declared provenance:
#   SOA 2014 Post-Level Term Lapse and Mortality Study.
#   RGA Reinsurance Company.
#   https://www.soa.org/resources/experience-studies/2014/
#   research-2014-post-level-shock/
#
# Bounded over D. No claim beyond D.

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List


# ---- Spread profile declaration ----------------------------------------

@dataclass
class SpreadProfile:
    """
    Declared characterization of relational spread across the duration field.

    Spread is the primary signal. Elevation ratio and peak location
    are the declared findings. No threshold. No detection frame.

    Open conditions:
        - level_term_period: declared as 10-year term; SOA data covers
          multiple term lengths — declare which cohort is in scope
        - baseline_window: pre-peak mean uses all durations before peak;
          sensitivity to window choice is an open condition
        - calendar_time: duration steps are not calendar time;
          translation requires declaration of timestep unit per cohort
    """
    level_term_period: int = 10         # declared for 10-year term
    peak_duration_index: Optional[int] = None    # index of maximum spread
    peak_spread_value: Optional[float] = None    # spread value at peak
    baseline_mean: Optional[float] = None        # mean spread pre-peak
    elevation_ratio: Optional[float] = None      # peak / baseline_mean
    spread_series: Optional[np.ndarray] = None   # full spread by duration

    provenance: str = (
        "SOA 2014 Post-Level Term Lapse and Mortality Study. "
        "Spread is observed in the lapse experience field (cell_lapse) "
        "per declared duration adjacency. Operators construct relational "
        "context. Finding lives in the lapse field, not the kernel output."
    )

    open_conditions: List[str] = field(default_factory=lambda: [
        "level_term_period: declared as 10-year term; verify cohort scope "
        "against SOA data before interpreting peak location",
        "baseline_window: pre-peak baseline uses all durations before peak; "
        "sensitivity to window choice not yet characterized",
        "calendar_time: duration steps are not calendar time; "
        "translation requires declaration of timestep unit per cohort",
        "column_names: SOA Excel column names must be verified against "
        "actual file before running on real data",
    ])

    def summary(self, durations: Optional[List] = None) -> str:
        lines = ["=== SPREAD PROFILE ==="]
        lines.append(f"Level term period declared: {self.level_term_period} years")
        lines.append("")
        if self.peak_duration_index is not None:
            dur_label = (
                f" (duration {durations[self.peak_duration_index]})"
                if durations else ""
            )
            lines += [
                "Relational spread — primary finding:",
                f"  Peak index:      {self.peak_duration_index}{dur_label}",
                f"  Peak value:      {self.peak_spread_value:.6f}",
                f"  Baseline mean:   {self.baseline_mean:.6f}  "
                f"(pre-peak durations)",
            ]
            if self.elevation_ratio is not None:
                lines.append(
                    f"  Elevation ratio: {self.elevation_ratio:.1f}x above baseline"
                )
        else:
            lines.append("Spread peak not determined — insufficient data.")
        lines.append("")
        lines.append("Provenance:")
        lines.append(f"  {self.provenance}")
        lines.append("")
        lines.append("Open conditions:")
        for oc in self.open_conditions:
            lines.append(f"  - {oc}")
        return "\n".join(lines)


# ---- Momentum characterization -----------------------------------------

@dataclass
class MomentumCharacterization:
    """
    Secondary characterization of directional trend over spread series.

    Momentum does not gate findings. It characterizes whether spread
    elevation was preceded by a rising directional signal.

    What is not claimed:
        - Momentum threshold is not a detection criterion
        - Momentum above threshold does not constitute a separate finding
        - Momentum is bounded by C operator — magnitude beyond unit scale discarded
    """
    peak_momentum_index: Optional[int] = None
    peak_momentum_value: Optional[float] = None
    momentum_series: Optional[np.ndarray] = None

    not_claimed: List[str] = field(default_factory=lambda: [
        "Momentum threshold is not a detection criterion",
        "Momentum result is secondary to spread elevation finding",
        "Magnitude beyond unit scale — discarded by C operator",
    ])

    def summary(self, durations: Optional[List] = None) -> str:
        lines = ["=== MOMENTUM CHARACTERIZATION (secondary) ==="]
        if self.peak_momentum_index is not None:
            dur_label = (
                f" (duration {durations[self.peak_momentum_index]})"
                if durations else ""
            )
            lines += [
                f"  Peak momentum index: {self.peak_momentum_index}{dur_label}",
                f"  Peak momentum value: {self.peak_momentum_value:.6f}",
            ]
        else:
            lines.append("  Momentum peak not determined.")
        lines.append("")
        lines.append("Not claimed:")
        for nc in self.not_claimed:
            lines.append(f"  - {nc}")
        return "\n".join(lines)


# ---- Spread characterization -------------------------------------------

def characterize_spread(spread_by_duration: np.ndarray,
                        durations: List,
                        level_term_period: int = 10) -> SpreadProfile:
    """
    Characterize relational spread across the declared duration field.

    Computes peak location, baseline mean, and elevation ratio.
    No threshold. Spread is observed and reported.

    Parameters
    ----------
    spread_by_duration : relational spread per duration (from detection.py)
    durations          : ordered duration values from data
    level_term_period  : declared level term length (default 10)

    Returns
    -------
    SpreadProfile with declared findings and open conditions.
    """
    profile = SpreadProfile(
        level_term_period=level_term_period,
        spread_series=spread_by_duration,
    )

    valid_mask = ~np.isnan(spread_by_duration)
    if valid_mask.sum() < 2:
        return profile

    peak_idx = int(np.nanargmax(spread_by_duration))
    peak_val = float(spread_by_duration[peak_idx])

    # Baseline: mean of all pre-peak durations with valid spread
    pre_peak = spread_by_duration[:peak_idx]
    pre_peak_valid = pre_peak[~np.isnan(pre_peak)]
    baseline = float(np.mean(pre_peak_valid)) if len(pre_peak_valid) > 0 else None

    profile.peak_duration_index = peak_idx
    profile.peak_spread_value = peak_val
    profile.baseline_mean = baseline
    if baseline and baseline > 0:
        profile.elevation_ratio = peak_val / baseline

    return profile


# ---- Momentum characterization -----------------------------------------

def characterize_momentum(momentum: np.ndarray,
                           durations: List) -> MomentumCharacterization:
    """
    Characterize directional trend over the spread series.

    Secondary characterization only. Does not gate findings.

    Parameters
    ----------
    momentum  : relational momentum series (from detection.py)
    durations : ordered duration values from data

    Returns
    -------
    MomentumCharacterization with secondary findings.
    """
    char = MomentumCharacterization(momentum_series=momentum)

    valid_mask = ~np.isnan(momentum)
    if valid_mask.sum() == 0:
        return char

    peak_idx = int(np.nanargmax(momentum))
    char.peak_momentum_index = peak_idx
    char.peak_momentum_value = float(momentum[peak_idx])

    return char


# ---- Full metrics pipeline ---------------------------------------------

def run_metrics(spread_by_duration: np.ndarray,
                momentum: np.ndarray,
                durations: List,
                level_term_period: int = 10,
                verbose: bool = True) -> "tuple[SpreadProfile, MomentumCharacterization]":
    """
    Full metrics pipeline: characterize spread → characterize momentum.

    Primary output: SpreadProfile — peak location, elevation ratio,
    and full spread series with declared provenance.

    Secondary output: MomentumCharacterization — directional trend
    over spread field. Does not gate primary findings.

    Parameters
    ----------
    spread_by_duration : relational spread per duration (from detection.py)
    momentum           : relational momentum series (from detection.py)
    durations          : ordered duration values from data
    level_term_period  : declared level term length (default 10)
    verbose            : print characterization summaries

    Returns
    -------
    (SpreadProfile, MomentumCharacterization)
    """
    # Primary: spread characterization
    profile = characterize_spread(
        spread_by_duration, durations, level_term_period
    )

    # Secondary: momentum characterization
    mom_char = characterize_momentum(momentum, durations)

    if verbose:
        print(profile.summary(durations))
        print()
        print(mom_char.summary(durations))

    return profile, mom_char
