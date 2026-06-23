# detection.py — Detection pipeline
#
# Connects declaration → relations → operators → signal.
#
# Pipeline:
#   1. Verify session declaration (Origin declares before operators act)
#   2. Load SOA data and verify domain membership
#   3. Build declared relational structure (typed edges)
#   4. Apply A → B → R → E over declared relations
#   5. Compute relational spread and momentum as detection signals
#   6. Return declared outputs with provenance
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
    r_field: Optional[np.ndarray] = None          # (n_edges,) kernel output E(x,ρ) = R(B(A(x)),ρ(A(x)))
    e_field: Optional[np.ndarray] = None          # (n_edges,) C(r) — declared projection of kernel output
    rho: Optional[np.ndarray] = None              # (n_cells,) local contrast

    # Detection signals — declared projections
    spread_by_duration: Optional[np.ndarray] = None   # relational spread per duration
    momentum: Optional[np.ndarray] = None              # relational momentum series

    # Declared projection discards
    projection_discards: list = field(default_factory=lambda: [
        "spread_by_duration: discards within-duration cell arrangement, "
        "retains scalar variance per duration step",
        "momentum: discards spread magnitude beyond unit scale via C, "
        "retains directional trend signal",
        "e_field: discards absolute lapse level (pre-A), "
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
            "Declared projection discards:",
        ]
        for d in self.projection_discards:
            lines.append(f"  - {d}")
        if self.open_conditions:
            lines.append("\nOpen conditions carried forward:")
            for c in self.open_conditions:
                lines.append(f"  [{c.status.value.upper()}] {c.name}: {c.declaration}")
        if self.spread_by_duration is not None:
            lines += [
                "",
                "Relational spread by duration:",
                f"  min:  {np.nanmin(self.spread_by_duration):.6f}",
                f"  max:  {np.nanmax(self.spread_by_duration):.6f}",
                f"  mean: {np.nanmean(self.spread_by_duration):.6f}",
            ]
        if self.momentum is not None:
            valid = self.momentum[~np.isnan(self.momentum)]
            if len(valid) > 0:
                lines += [
                    "",
                    "Relational momentum:",
                    f"  min:  {valid.min():.6f}",
                    f"  max:  {valid.max():.6f}",
                    f"  mean: {valid.mean():.6f}",
                ]
        return "\n".join(lines)


# ---- Data loader -------------------------------------------------------

def load_soa_data(filepath: str,
                  sheet_name: str = "Lapse Study",
                  lapse_col: str = "LapseRate",
                  cohort_col: str = "IssueYear",
                  duration_col: str = "Duration",
                  jump_col: str = "PremiumJumpBand") -> pd.DataFrame:
    """
    Load SOA Post-Level Term lapse study data from Excel.

    Column names are declared open condition — verify against
    actual file and update parameters if needed.

    Parameters
    ----------
    filepath     : path to SOA Excel file
    sheet_name   : worksheet name (open condition — verify)
    lapse_col    : lapse rate column name (open condition — verify)
    cohort_col   : issue year column name (open condition — verify)
    duration_col : duration column name (open condition — verify)
    jump_col     : premium jump band column name (open condition — verify)
    """
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
                  verbose: bool = True) -> DetectionResult:
    """
    Full detection pipeline: declare → load → relate → detect.

    Parameters
    ----------
    filepath        : path to SOA Excel file
    rho_base        : local contrast scaling (default 0.4, per kernel spec)
    momentum_window : rolling window for momentum signal
    include_jump    : include premium jump adjacency (default True)
    verbose         : print declaration and result summary

    Returns
    -------
    DetectionResult with all declared outputs and open conditions.
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
    # c_projected is C(r) — bounded coherence applied as final projection.
    # Preserves: sign and ordering of relational circulation field.
    # Discards: magnitude beyond unit scale.
    c_projected = operator_c(r)

    # ---- Step 6: Relational spread by duration -------------------------
    # Declared projection: one spread value per duration step.
    # Preserves: scalar relational variance across cells at each duration.
    # Discards: within-duration cell arrangement.
    durations = sorted(df[duration_col].unique())
    spread_series = np.full(len(durations), np.nan)

    for i, dur in enumerate(durations):
        dur_mask = df[df[duration_col] == dur]['cell_idx'].values
        if len(dur_mask) > 1:
            dur_field = cell_lapse[dur_mask]
            spread_series[i] = relational_spread(dur_field)

    # ---- Step 7: Relational momentum -----------------------------------
    # Declared projection: bounded trend signal over spread series.
    # Preserves: directional asymmetry in spread trajectory.
    # Discards: magnitude beyond unit scale (C operator).
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
        r_field=r,                  # kernel output E(x,ρ) = R(B(A(x)),ρ(A(x)))
        e_field=c_projected,        # C(r) — declared projection of kernel output
        rho=rho,
        spread_by_duration=spread_series,
        momentum=momentum,
        open_conditions=open_conditions,
    )

    if verbose:
        print(result.summary())

    return result, durations
