# detection.py — Detection pipeline
#
# Connects declaration → relations → operators → signal.
#
# Pipeline:
#   1. Verify session declaration (Origin declares before operators act)
#   2. Load SOA data and verify domain membership
#   3. Build declared relational structure (typed edges)
#   4. Apply A → B → R → E over declared relations
#   5. Compute relational spread by duration as the primary signal
#   6. Compute relational momentum as secondary characterization signal
#   7. Return declared outputs with provenance
#
# Primary goal: observe and characterize relational spread across the
# duration field. Spread elevation is the finding. No threshold-as-detection.
#
# No operator is called before declaration.verify() succeeds.
# No silent reduction. All projections declared.
# Bounded over D. No claim beyond D.

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from .declaration import SessionDeclaration, AdmissibilityStatus
from .operators import (
    DeclaredRelations, EdgeType,
    operator_a, operator_b, operator_r, operator_e, operator_c,
    compute_rho, relational_spread, relational_momentum
)
from .relations import build_cell_index, build_declared_relations


# ---- Declared output ---------------------------------------------------

@dataclass
class DetectionResult:
    """
    Declared output of the detection pipeline.

    Primary signal: relational spread by duration.
    Spread is observed and reported — not thresholded.
    Elevation ratio and peak location are declared findings.

    All fields state what they are, what they preserve,
    and what they discard — per C constraint.
    """
    # Input provenance
    n_cells: int = 0
    n_edges_duration: int = 0
    n_edges_jump: int = 0
    n_durations: int = 0

    # Field outputs
    lapse_field: Optional[np.ndarray] = None     # (n_cells,) raw lapse rates
    a_field: Optional[np.ndarray] = None          # (n_edges,) gradient field A(x)
    b_field: Optional[np.ndarray] = None          # (n_edges,) accumulated field B(A(x))
    e_kernel: Optional[np.ndarray] = None         # (n_edges,) kernel output E(x,ρ) = R(B(A(x)),ρ(A(x)))
    c_projection: Optional[np.ndarray] = None     # (n_edges,) C(E) — declared projection of kernel output
    rho: Optional[np.ndarray] = None              # (n_cells,) local contrast

    # Primary signal — relational spread by duration
    spread_by_duration: Optional[np.ndarray] = None   # (n_durations,) spread per duration step

    # Secondary signal — relational momentum over spread series
    momentum: Optional[np.ndarray] = None             # (n_durations,) directional trend signal

    # Declared projection discards
    projection_discards: list = field(default_factory=lambda: [
        "spread_by_duration: discards within-duration cell arrangement, "
        "retains scalar relational variance per duration step",
        "momentum: discards spread magnitude beyond unit scale via C, "
        "retains directional trend signal as secondary characterization",
        "c_projection: discards absolute lapse level (pre-A), "
        "retains relational gradient structure",
    ])

    # Open conditions carried forward from declaration
    open_conditions: list = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=== DETECTION RESULT ===",
            f"Cells:           {self.n_cells}",
            f"Duration edges:  {self.n_edges_duration}",
            f"Jump edges:      {self.n_edges_jump}",
            f"Duration steps:  {self.n_durations}",
            "",
            "Primary signal: relational spread by duration",
            "Spread is observed and reported — not thresholded.",
            "",
            "Declared projection discards:",
        ]
        for d in self.projection_discards:
            lines.append(f"  - {d}")
        if self.open_conditions:
            lines.append("\nOpen conditions carried forward:")
            for c in self.open_conditions:
                lines.append(f"  [{c.status.value.upper()}] {c.name}: {c.declaration}")
        if self.spread_by_duration is not None:
            peak_idx = int(np.nanargmax(self.spread_by_duration))
            baseline = np.nanmean(self.spread_by_duration[:peak_idx]) if peak_idx > 0 else np.nan
            peak_val = float(np.nanmax(self.spread_by_duration))
            elevation = peak_val / baseline if (baseline and baseline > 0) else None
            lines += [
                "",
                "Relational spread by duration:",
                f"  min:       {np.nanmin(self.spread_by_duration):.6f}",
                f"  max:       {peak_val:.6f}  (at duration index {peak_idx})",
                f"  mean:      {np.nanmean(self.spread_by_duration):.6f}",
            ]
            if elevation is not None:
                lines.append(
                    f"  elevation: {elevation:.1f}x above pre-peak baseline mean"
                )
        if self.momentum is not None:
            valid = self.momentum[~np.isnan(self.momentum)]
            if len(valid) > 0:
                lines += [
                    "",
                    "Relational momentum (secondary characterization):",
                    f"  min:  {valid.min():.6f}",
                    f"  max:  {valid.max():.6f}",
                    f"  mean: {valid.mean():.6f}",
                ]
        return "\n".join(lines)


# ---- Data loader -------------------------------------------------------

def load_soa_data(filepath: str,
                  sheet_name: Optional[str] = "Lapse Study",
                  lapse_col: str = "LapseRate",
                  cohort_col: str = "IssueYear",
                  duration_col: str = "Duration",
                  jump_col: str = "PremiumJumpBand") -> pd.DataFrame:
    """
    Load lapse study data from Excel or CSV.

    Accepts:
      - SOA Excel file (sheet_name required)
      - Synthetic CSV (sheet_name ignored)

    Column names are declared open condition for Excel.
    Synthetic CSV uses declared column names by construction.
    """
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath, sheet_name=sheet_name)

    required = [lapse_col, cohort_col, duration_col, jump_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Column names not found: {missing}\n"
            f"Available columns: {list(df.columns)}\n"
            f"Update column name parameters to match actual SOA file. "
            f"This is declared open condition 'column_names' in declaration.py."
        )

    df = df[required].dropna(subset=[lapse_col])
    df[lapse_col] = df[lapse_col].astype(float)
    return df


# ---- Main pipeline -----------------------------------------------------

def run_detection(filepath: str,
                  sheet_name: str = "Lapse Study",
                  lapse_col: str = "LapseRate",
                  cohort_col: str = "IssueYear",
                  duration_col: str = "Duration",
                  jump_col: str = "PremiumJumpBand",
                  rho_base: float = 0.4,
                  momentum_window: int = 5,
                  include_jump: bool = True,
                  verbose: bool = True) -> "tuple[DetectionResult, list]":
    """
    Full detection pipeline: declare → load → relate → detect.

    Primary output: relational spread by duration.
    Spread elevation is the finding — observed and reported, not thresholded.

    Parameters
    ----------
    filepath        : path to SOA Excel file or synthetic CSV
    rho_base        : local contrast scaling (default 0.4, per kernel spec)
    momentum_window : rolling window for momentum signal (secondary)
    include_jump    : include premium jump adjacency (default True)
    verbose         : print declaration and result summary

    Returns
    -------
    (DetectionResult, durations) with all declared outputs and open conditions.
    """

    # ---- Step 1: Declaration (Origin declares before operators act) ----
    decl = SessionDeclaration()
    decl.verify()   # raises if any condition is INADMISSIBLE
    if verbose:
        print(decl.report())
        print()

    open_conditions = decl.mapping.open_conditions()

    # ---- Step 2: Load and verify domain membership ---------------------
    df = load_soa_data(
        filepath, sheet_name, lapse_col, cohort_col, duration_col, jump_col
    )

    lapse_values = df[lapse_col].values
    if not decl.domain.verify(lapse_values):
        raise ValueError(
            f"Data fails domain verification. "
            f"Expected values in [{decl.domain.lower_bound}, "
            f"{decl.domain.upper_bound}], finite. "
            f"Check lapse rate column: {lapse_col}"
        )

    # ---- Step 3: Build cell index and declared relations ---------------
    df, cells = build_cell_index(df, cohort_col, duration_col, jump_col)

    rel = build_declared_relations(
        cells,
        include_duration=True,
        include_jump=include_jump,
        include_cohort=False,       # open condition — excluded
        cohort_col=cohort_col,
        duration_col=duration_col,
        jump_col=jump_col
    )

    n_dur = len(rel.edges_of_type(EdgeType.DURATION))
    n_jmp = len(rel.edges_of_type(EdgeType.JUMP))

    # ---- Step 4: Build lapse field over declared cells -----------------
    # Aggregate to one lapse rate per cell (mean if multiple rows per cell)
    cell_lapse = (
        df.groupby('cell_idx')[lapse_col]
        .mean()
        .reindex(range(rel.n_nodes))
        .fillna(0.0)
        .values
    )

    # ---- Step 5: Apply ABRCE operators ---------------------------------
    a = operator_a(cell_lapse, rel)
    rho = compute_rho(a, rel, rho_base)
    b = operator_b(a, rel)
    r = operator_r(b, rel, rho)
    # C is a declared projection of the kernel output, not a kernel operator.
    # r is the kernel output E(x,ρ) = R(B(A(x)),ρ(A(x))).
    # c_projected is C(E) — bounded coherence applied as final projection.
    # Preserves: sign and ordering of relational circulation field.
    # Discards: magnitude beyond unit scale.
    c_projected = operator_c(r)

    # ---- Step 6: Relational spread by duration (PRIMARY SIGNAL) --------
    # Declared by Origin: observable is spread in lapse experience field.
    # Operators construct relational context. Finding lives in cell_lapse.
    # No threshold. No detection frame. Elevation is the finding.
    #
    # Declared projection:
    #   Preserves: scalar relational variance across cells at each duration.
    #   Discards: within-duration cell arrangement.
    durations = sorted(df[duration_col].unique())
    spread_series = np.full(len(durations), np.nan)

    for i, dur in enumerate(durations):
        dur_mask = df[df[duration_col] == dur]['cell_idx'].values
        if len(dur_mask) > 1:
            dur_field = cell_lapse[dur_mask]
            spread_series[i] = relational_spread(dur_field)

    # ---- Step 7: Relational momentum (SECONDARY SIGNAL) ----------------
    # Momentum characterizes the directional trend over the spread series.
    # Secondary signal — does not gate or replace spread as primary finding.
    #
    # Declared projection:
    #   Preserves: directional asymmetry in spread trajectory.
    #   Discards: magnitude beyond unit scale (C operator).
    momentum = relational_momentum(
        spread_series, window=momentum_window, rho_base=rho_base
    )

    # ---- Assemble declared result --------------------------------------
    result = DetectionResult(
        n_cells=rel.n_nodes,
        n_edges_duration=n_dur,
        n_edges_jump=n_jmp,
        n_durations=len(durations),
        lapse_field=cell_lapse,
        a_field=a,
        b_field=b,
        e_kernel=r,                  # kernel output E(x,ρ) = R(B(A(x)),ρ(A(x)))
        c_projection=c_projected,    # C(E) — declared projection of kernel output
        rho=rho,
        spread_by_duration=spread_series,
        momentum=momentum,
        open_conditions=open_conditions,
    )

    if verbose:
        print(result.summary())

    return result, durations
