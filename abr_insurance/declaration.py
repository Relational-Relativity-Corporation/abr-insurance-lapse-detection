# declaration.py — Domain and measurement mapping declaration
#
# This file declares D, M, and the admissibility conditions for the
# ABR insurance lapse detection analysis before any operator acts.
#
# Under the ABR framework, nothing is evaluated before something is
# declared. This file is the declaration. It must be executed and
# its conditions verified before any operator in operators.py is called.
#
# Bounded over D. No claim beyond D.

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


# ---- Admissibility status ----------------------------------------------

class AdmissibilityStatus(Enum):
    ADMISSIBLE     = "admissible"
    OPEN_CONDITION = "open_condition"   # declared but not yet resolved
    INADMISSIBLE   = "inadmissible"     # declared and known to fail


@dataclass
class AdmissibilityCondition:
    name: str
    status: AdmissibilityStatus
    declaration: str
    constraint: Optional[str] = None    # what would make it inadmissible
    resolution: Optional[str] = None    # how open condition would be resolved


# ---- Domain declaration ------------------------------------------------

@dataclass
class DomainDeclaration:
    """
    D := { x ∈ ℝⁿ | n < ∞, |x[i]| < ∞ ∀ i }

    Declared bounded domain for the SOA lapse experience analysis.
    All elements are finite real numbers. No claim beyond D.
    """
    name: str = "SOA Post-Level Term Lapse Experience Domain"
    description: str = (
        "Finite-dimensional field of lapse rates over SOA experience cells. "
        "Each element is a per-cell observed lapse rate — a finite real number "
        "in [0, 1]. The field is bounded by construction."
    )
    lower_bound: float = 0.0    # lapse rate floor
    upper_bound: float = 1.0    # lapse rate ceiling
    element_type: str = "per-cell observed lapse rate"

    def verify(self, values) -> bool:
        """Verify all values are within declared domain bounds."""
        import numpy as np
        arr = np.asarray(values, dtype=float)
        in_bounds = bool(np.all(arr >= self.lower_bound) and
                        np.all(arr <= self.upper_bound))
        finite = bool(np.all(np.isfinite(arr)))
        return in_bounds and finite


# ---- Measurement mapping declaration -----------------------------------

@dataclass
class MeasurementMapping:
    """
    M : O → D

    Declares how the observable (SOA lapse study data) is mapped
    into the declared domain D. Constraints on M are declared here
    before any operator acts on M(o).
    """
    source: str = (
        "SOA 2014 Post-Level Term Lapse and Mortality Study. "
        "Society of Actuaries / RGA Reinsurance Company. "
        "https://www.soa.org/resources/experience-studies/2014/"
        "research-2014-post-level-shock/"
    )
    observable: str = (
        "Aggregated lapse experience for US level premium term life "
        "insurance policies, 2000-2012, across multiple companies."
    )
    mapping: str = (
        "Each experience cell maps to a single lapse rate value: "
        "observed lapses / exposed policies within the cell definition. "
        "Cell definition: study year × duration × gender × issue age × "
        "face amount band × premium jump band."
    )

    # Declared constraints on M
    constraints: List[AdmissibilityCondition] = field(default_factory=lambda: [

        AdmissibilityCondition(
            name="aggregation_loss",
            status=AdmissibilityStatus.ADMISSIBLE,
            declaration=(
                "Data is aggregated across multiple companies. "
                "Individual policy-level relational structure has been "
                "partially discarded through aggregation."
            ),
            constraint=(
                "Cell-level aggregation preserves within-cell summary "
                "statistics but discards between-policy relational structure. "
                "This is declared and accepted for this analysis."
            ),
            resolution=(
                "Policy-level data from Symetra would resolve this condition "
                "and allow full relational analysis at the policy level."
            )
        ),

        AdmissibilityCondition(
            name="hub_topology",
            status=AdmissibilityStatus.ADMISSIBLE,
            declaration=(
                "All data derives from a single primary source (SOA/RGA). "
                "Apparent multi-company diversity does not constitute "
                "independent measurement mapping — companies contribute "
                "experience to a pooled study under SOA methodology."
            ),
            constraint=(
                "Results reflect industry aggregate experience, not "
                "Symetra-specific experience. Symetra may deviate from "
                "industry aggregate in ways not visible in this data."
            ),
            resolution=(
                "Symetra-specific experience data would provide an "
                "independent M with distinct incentive structure."
            )
        ),

        AdmissibilityCondition(
            name="premium_jump_structure",
            status=AdmissibilityStatus.ADMISSIBLE,
            declaration=(
                "Premium jump bands classify cells by magnitude of premium "
                "increase at end of level term — a single end-of-term event. "
                "Verified against Symetra product documentation: premiums are "
                "level throughout the initial term period. Jump is not a "
                "mid-policy phenomenon."
            ),
            constraint=(
                "Jump band adjacency is a cross-sectional relation at the "
                "shock point only. It does not represent temporal continuation "
                "and does not participate in B accumulation."
            )
        ),

        AdmissibilityCondition(
            name="column_names",
            status=AdmissibilityStatus.OPEN_CONDITION,
            declaration=(
                "SOA Excel column names are assumed from study description. "
                "Actual column names have not been verified against the "
                "downloaded file."
            ),
            constraint=(
                "If actual column names differ from assumed names "
                "(IssueYear, Duration, PremiumJumpBand, LapseRate), "
                "the relations builder will fail or produce incorrect results."
            ),
            resolution=(
                "Download SOA zip, open Excel, verify column names, "
                "update relations.py and detection.py accordingly."
            )
        ),

        AdmissibilityCondition(
            name="cohort_adjacency",
            status=AdmissibilityStatus.OPEN_CONDITION,
            declaration=(
                "Issue cohort adjacency (neighboring issue years) is "
                "structurally available but not declared for use. "
                "What constitutes meaningful cohort coupling in this "
                "dataset has not been declared."
            ),
            constraint=(
                "Cohort adjacency is excluded from all operator calls "
                "until this condition is resolved."
            ),
            resolution=(
                "Declare what relational claim cohort adjacency is making "
                "and what provenance supports it before enabling."
            )
        ),

        AdmissibilityCondition(
            name="benchmark_equivalence",
            status=AdmissibilityStatus.OPEN_CONDITION,
            declaration=(
                "The relational momentum signal is compared against SOA "
                "reported shock lapse timing. The SOA study identifies "
                "shock lapse at end of level term as the primary event. "
                "Whether relational momentum fires before, at, or after "
                "the SOA-identified shock lapse point is the empirical "
                "question this analysis addresses."
            ),
            constraint=(
                "Lead time claim requires empirical demonstration, not "
                "assertion. No lead time claim is made before the analysis "
                "is run and results are declared."
            ),
            resolution=(
                "Run experiments/run_analysis.py on SOA data and declare "
                "the measured result."
            )
        ),

    ])

    def admissible(self) -> bool:
        """Returns True only if no INADMISSIBLE conditions exist."""
        return all(
            c.status != AdmissibilityStatus.INADMISSIBLE
            for c in self.constraints
        )

    def open_conditions(self) -> List[AdmissibilityCondition]:
        return [c for c in self.constraints
                if c.status == AdmissibilityStatus.OPEN_CONDITION]

    def report(self) -> str:
        lines = [
            "=== MEASUREMENT MAPPING DECLARATION ===",
            f"Source:     {self.source}",
            f"Observable: {self.observable}",
            f"Mapping:    {self.mapping}",
            "",
            "=== ADMISSIBILITY CONDITIONS ===",
        ]
        for c in self.constraints:
            lines.append(f"\n[{c.status.value.upper()}] {c.name}")
            lines.append(f"  Declaration: {c.declaration}")
            if c.constraint:
                lines.append(f"  Constraint:  {c.constraint}")
            if c.resolution and c.status == AdmissibilityStatus.OPEN_CONDITION:
                lines.append(f"  Resolution:  {c.resolution}")

        open_c = self.open_conditions()
        lines += [
            "",
            "=== SUMMARY ===",
            f"Admissible:       {self.admissible()}",
            f"Open conditions:  {len(open_c)}",
        ]
        if open_c:
            lines.append("Open condition names: " +
                        ", ".join(c.name for c in open_c))
        return "\n".join(lines)


# ---- Full session declaration ------------------------------------------

@dataclass
class SessionDeclaration:
    """
    Complete Origin declaration for this analysis session.
    Must be instantiated and verified before any operator is called.

    Usage:
        decl = SessionDeclaration()
        decl.verify()   # raises if inadmissible
        print(decl.report())
    """
    domain: DomainDeclaration = field(
        default_factory=DomainDeclaration)
    mapping: MeasurementMapping = field(
        default_factory=MeasurementMapping)

    # Declared relational scope for this session
    relations_declared: List[str] = field(default_factory=lambda: [
        "DURATION — temporal continuation along policy year path",
        "JUMP     — cross-sectional shock magnitude adjacency at end of term",
    ])
    relations_excluded: List[str] = field(default_factory=lambda: [
        "COHORT   — open condition, excluded pending declaration",
        "RING     — proof construct only, never admissible in application",
    ])

    # Declared detection target
    detection_target: str = (
        "Relational momentum signal derived from ABRCE operators over "
        "declared cell adjacency, compared against SOA-reported shock "
        "lapse timing at end of level term period."
    )

    # What is discarded — C constraint applied to the session
    discarded: List[str] = field(default_factory=lambda: [
        "Individual policy-level relational structure (aggregated away in SOA data)",
        "Company-specific experience (pooled across multiple companies)",
        "Cohort adjacency (open condition)",
        "Magnitude beyond unit scale (C operator projection)",
    ])

    def verify(self):
        """Raise if any INADMISSIBLE condition exists."""
        if not self.mapping.admissible():
            inadmissible = [
                c.name for c in self.mapping.constraints
                if c.status == AdmissibilityStatus.INADMISSIBLE
            ]
            raise ValueError(
                f"Session inadmissible. Failed conditions: {inadmissible}"
            )

    def report(self) -> str:
        lines = [
            "=== SESSION DECLARATION ===",
            "",
            f"Domain: {self.domain.name}",
            f"  {self.domain.description}",
            "",
            "Relations declared:",
        ]
        for r in self.relations_declared:
            lines.append(f"  + {r}")
        lines.append("Relations excluded:")
        for r in self.relations_excluded:
            lines.append(f"  - {r}")
        lines += [
            "",
            f"Detection target: {self.detection_target}",
            "",
            "Discarded (declared per C constraint):",
        ]
        for d in self.discarded:
            lines.append(f"  - {d}")
        lines += [
            "",
            self.mapping.report(),
        ]
        return "\n".join(lines)


# ---- Entry point -------------------------------------------------------

if __name__ == "__main__":
    decl = SessionDeclaration()
    decl.verify()
    print(decl.report())
