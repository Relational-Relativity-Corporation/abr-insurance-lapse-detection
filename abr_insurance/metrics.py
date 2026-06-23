# metrics.py — Lead time computation vs SOA shock lapse benchmark
#
# Compares relational momentum signal against SOA-reported shock lapse
# timing at end of level term period.
#
# Declared benchmark:
#   The SOA 2014 Post-Level Term study identifies shock lapse as the
#   primary risk event at the transition from level term to annually
#   renewable term (ART). Shock lapse is characterized by a sharp
#   elevation in lapse rates at the first post-level duration.
#
# What this file measures:
#   Whether relational momentum fires before, at, or after the
#   SOA-identified shock lapse point — and by how many duration steps.
#
# What this file does not claim:
#   No lead time claim is made before results are declared.
#   The benchmark is the SOA shock lapse point, not a simulated failure.
#   Duration steps are not calendar time — translation requires
#   declaration of the timestep unit for each product cohort.
#
# Bounded over D. No claim beyond D.

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple


# ---- Benchmark declaration ---------------------------------------------

@dataclass
class ShockLapseBenchmark:
    """
    Declared benchmark: SOA shock lapse point.

    The SOA study identifies shock lapse at the first post-level duration
    — the duration immediately following end of level term period.
    For a 10-year level term, shock lapse occurs at duration 11 (PLT year 1).

    This is the comparison point for relational momentum detection.

    Declared provenance:
        SOA 2014 Post-Level Term Lapse and Mortality Study.
        RGA Reinsurance Company.
        https://www.soa.org/resources/experience-studies/2014/
        research-2014-post-level-shock/

    Open condition:
        The exact duration index of shock lapse in the SOA Excel data
        depends on how durations are indexed in the actual file.
        Verify before interpreting lead time results.
    """
    level_term_period: int = 10          # years — declared for 10-year term
    shock_duration_index: Optional[int] = None  # set after data inspection
    shock_lapse_rate: Optional[float] = None    # observed in SOA data
    provenance: str = (
        "SOA 2014 Post-Level Term Lapse and Mortality Study. "
        "Shock lapse identified at first post-level duration (PLT year 1)."
    )
    open_conditions: List[str] = field(default_factory=lambda: [
        "shock_duration_index: verify exact duration index in SOA data "
        "before interpreting lead time",
        "level_term_period: declared as 10-year term; SOA data covers "
        "multiple term lengths — declare which cohort is in scope",
        "calendar_time: duration steps are not calendar time; "
        "translation requires declaration of timestep unit per cohort",
    ])


# ---- Detection trigger -------------------------------------------------

@dataclass
class TriggerResult:
    """
    Declared result of momentum signal detection vs benchmark.

    All fields state what they are and what they do not claim.
    """
    # Signal detection
    signal_trigger_index: Optional[int] = None    # duration index when signal fires
    signal_trigger_value: Optional[float] = None  # momentum value at trigger
    signal_threshold: float = 0.10                # declared threshold

    # Benchmark
    shock_duration_index: Optional[int] = None    # SOA shock lapse duration index

    # Lead time — declared open condition until results confirmed
    lead_time_steps: Optional[int] = None         # steps before shock (positive = early)
    lead_time_declared: bool = False               # True only after empirical confirmation

    # What is not claimed
    not_claimed: List[str] = field(default_factory=lambda: [
        "Lead time in calendar time — duration steps only",
        "Universality — result is bounded to this dataset and declaration",
        "Causality — signal correlation with shock lapse, not causation",
        "Company-specific applicability — SOA data is industry aggregate",
    ])

    def summary(self) -> str:
        lines = ["=== TRIGGER RESULT ==="]
        if self.signal_trigger_index is not None:
            lines += [
                f"Signal fires at duration index: {self.signal_trigger_index}",
                f"Momentum value at trigger:      {self.signal_trigger_value:.4f}",
                f"Signal threshold:               {self.signal_threshold}",
            ]
        else:
            lines.append("Signal did not fire above threshold in this dataset.")

        if self.shock_duration_index is not None:
            lines.append(
                f"SOA shock lapse duration index: {self.shock_duration_index}"
            )

        if self.lead_time_steps is not None:
            qualifier = "(declared)" if self.lead_time_declared else "(provisional)"
            sign = "before" if self.lead_time_steps > 0 else \
                   "after" if self.lead_time_steps < 0 else "at"
            lines.append(
                f"Lead time: {abs(self.lead_time_steps)} duration steps "
                f"{sign} shock lapse {qualifier}"
            )
        else:
            lines.append("Lead time: not computed — shock index not declared.")

        lines.append("\nNot claimed:")
        for nc in self.not_claimed:
            lines.append(f"  - {nc}")
        return "\n".join(lines)


# ---- Trigger computation -----------------------------------------------

def compute_trigger(momentum: np.ndarray,
                    durations: List,
                    benchmark: ShockLapseBenchmark,
                    signal_threshold: float = 0.10) -> TriggerResult:
    """
    Compute momentum signal trigger and lead time vs SOA benchmark.

    Parameters
    ----------
    momentum         : relational momentum series (one value per duration)
    durations        : ordered list of duration values from data
    benchmark        : declared SOA shock lapse benchmark
    signal_threshold : momentum threshold for detection declaration

    Returns
    -------
    TriggerResult with trigger index, shock index, and lead time.
    Lead time is provisional until benchmark.shock_duration_index is
    verified against actual SOA data.
    """
    result = TriggerResult(
        signal_threshold=signal_threshold,
        shock_duration_index=benchmark.shock_duration_index,
    )

    # Find first duration where momentum exceeds threshold
    trigger_idx = next(
        (i for i, v in enumerate(momentum)
         if not np.isnan(v) and v >= signal_threshold),
        None
    )

    if trigger_idx is not None:
        result.signal_trigger_index = trigger_idx
        result.signal_trigger_value = float(momentum[trigger_idx])

    # Compute lead time if benchmark shock index is declared
    if trigger_idx is not None and benchmark.shock_duration_index is not None:
        lead = benchmark.shock_duration_index - trigger_idx
        result.lead_time_steps = lead
        # Provisional until shock_duration_index verified against actual data
        result.lead_time_declared = False

    return result


# ---- Shock lapse detection from data -----------------------------------

def detect_shock_lapse(spread_by_duration: np.ndarray,
                       durations: List,
                       multiplier: float = 2.0) -> Tuple[Optional[int], Optional[float]]:
    """
    Identify shock lapse point from relational spread series.

    Shock lapse produces a sharp elevation in relational spread —
    the lapse rate field becomes highly non-uniform at the shock duration
    as some cells lapse at very high rates and others do not.

    Detection: first duration where spread exceeds multiplier * prior mean.

    Parameters
    ----------
    spread_by_duration : relational spread per duration step
    durations          : ordered duration values
    multiplier         : elevation factor (default 2.0 — declared open)

    Returns
    -------
    (shock_index, shock_spread_value) or (None, None) if not detected.

    Open condition: multiplier is declared but not calibrated to SOA data.
    Verify against SOA reported shock lapse magnitudes after data inspection.
    """
    valid = spread_by_duration[~np.isnan(spread_by_duration)]
    if len(valid) < 3:
        return None, None

    for i in range(2, len(spread_by_duration)):
        if np.isnan(spread_by_duration[i]):
            continue
        prior_mean = np.nanmean(spread_by_duration[:i])
        if prior_mean > 0 and spread_by_duration[i] >= multiplier * prior_mean:
            return i, float(spread_by_duration[i])

    return None, None


# ---- Full metrics pipeline ---------------------------------------------

def run_metrics(spread_by_duration: np.ndarray,
                momentum: np.ndarray,
                durations: List,
                level_term_period: int = 10,
                signal_threshold: float = 0.10,
                shock_multiplier: float = 2.0,
                soa_shock_duration_index: Optional[int] = None,
                verbose: bool = True) -> TriggerResult:
    """
    Full metrics pipeline: declare benchmark → compute trigger →
    compute lead time.

    Benchmark declaration:
        The SOA shock lapse point must be declared from the SOA study
        directly — not derived from ABR output. Pass soa_shock_duration_index
        after inspecting the SOA data and identifying the first post-level
        duration where shock lapse is reported.

        Until soa_shock_duration_index is declared, lead time is not computed.
        detect_shock_lapse() runs as secondary validation only — comparing
        ABR-detected spread elevation against the SOA-declared benchmark,
        not constructing it.

    Parameters
    ----------
    spread_by_duration      : relational spread per duration (from detection.py)
    momentum                : relational momentum series (from detection.py)
    durations               : ordered duration values from data
    level_term_period       : declared level term length (default 10)
    signal_threshold        : momentum detection threshold (default 0.10)
    shock_multiplier        : spread elevation factor for secondary validation
    soa_shock_duration_index: duration index of SOA-reported shock lapse.
                              Declared by Origin after SOA data inspection.
                              Required for lead time computation.

    Returns
    -------
    TriggerResult with all declared findings and open conditions.
    """
    # Declare benchmark from SOA — not from ABR output
    benchmark = ShockLapseBenchmark(level_term_period=level_term_period)

    if soa_shock_duration_index is not None:
        benchmark.shock_duration_index = soa_shock_duration_index
        if verbose:
            print(f"Benchmark declared: SOA shock lapse at duration index "
                  f"{soa_shock_duration_index} "
                  f"(duration {durations[soa_shock_duration_index]})\n")
    else:
        if verbose:
            print(
                "SOA shock duration index not declared.\n"
                "Inspect SOA data, identify first post-level duration with\n"
                "elevated lapse rate, then pass soa_shock_duration_index.\n"
                "Lead time will not be computed until this is declared.\n"
            )

    # Secondary validation: ABR spread elevation vs declared benchmark
    # This is not benchmark construction — it is a secondary check.
    abr_shock_idx, abr_shock_val = detect_shock_lapse(
        spread_by_duration, durations, multiplier=shock_multiplier
    )

    if verbose and abr_shock_idx is not None:
        if benchmark.shock_duration_index is not None:
            agreement = (abr_shock_idx == benchmark.shock_duration_index)
            print(
                f"Secondary validation: ABR spread elevation detected at "
                f"duration index {abr_shock_idx} "
                f"(duration {durations[abr_shock_idx]}), "
                f"spread = {abr_shock_val:.6f}\n"
                f"Agreement with SOA benchmark: {agreement}\n"
            )
        else:
            print(
                f"Secondary validation only (no SOA benchmark declared): "
                f"ABR spread elevation at duration index {abr_shock_idx}, "
                f"spread = {abr_shock_val:.6f}\n"
            )

    # Compute trigger and lead time against SOA-declared benchmark
    result = compute_trigger(
        momentum, durations, benchmark, signal_threshold
    )

    if verbose:
        print(result.summary())
        if benchmark.open_conditions:
            print("\nBenchmark open conditions:")
            for oc in benchmark.open_conditions:
                print(f"  - {oc}")

    return result
