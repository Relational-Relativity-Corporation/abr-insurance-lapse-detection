# tests/test_declaration.py — Admissibility logic tests
#
# Tests verify that:
# - Domain verification correctly bounds values
# - Admissibility logic raises on INADMISSIBLE conditions
# - Open conditions pass through without blocking
# - Session declaration carries open conditions correctly
# - Discarded information is declared explicitly
#
# No SOA file required. All tests use synthetic declarations.
# Bounded over D. No claim beyond D.

import pytest
import numpy as np
from abr_insurance.declaration import (
    DomainDeclaration,
    MeasurementMapping,
    SessionDeclaration,
    AdmissibilityCondition,
    AdmissibilityStatus,
)


# ---- DomainDeclaration tests -------------------------------------------

class TestDomainDeclaration:

    def test_valid_lapse_rates_pass(self):
        """Values in [0, 1] are within declared domain."""
        domain = DomainDeclaration()
        values = np.array([0.0, 0.01, 0.25, 0.5, 0.99, 1.0])
        assert domain.verify(values) is True

    def test_negative_value_fails(self):
        """Negative lapse rate is outside declared domain."""
        domain = DomainDeclaration()
        values = np.array([0.1, -0.01, 0.5])
        assert domain.verify(values) is False

    def test_above_unity_fails(self):
        """Lapse rate above 1.0 is outside declared domain."""
        domain = DomainDeclaration()
        values = np.array([0.5, 1.01])
        assert domain.verify(values) is False

    def test_nan_fails(self):
        """NaN values are outside declared domain."""
        domain = DomainDeclaration()
        values = np.array([0.1, np.nan, 0.5])
        assert domain.verify(values) is False

    def test_inf_fails(self):
        """Infinite values are outside declared domain."""
        domain = DomainDeclaration()
        values = np.array([0.1, np.inf, 0.5])
        assert domain.verify(values) is False

    def test_empty_array_passes(self):
        """Empty array is trivially in domain."""
        domain = DomainDeclaration()
        assert domain.verify(np.array([])) is True

    def test_boundary_values_pass(self):
        """Exact boundary values 0.0 and 1.0 are admissible."""
        domain = DomainDeclaration()
        assert domain.verify(np.array([0.0, 1.0])) is True


# ---- MeasurementMapping admissibility tests ----------------------------

class TestMeasurementMapping:

    def test_default_mapping_admissible(self):
        """Default measurement mapping has no INADMISSIBLE conditions."""
        mapping = MeasurementMapping()
        assert mapping.admissible() is True

    def test_open_conditions_present(self):
        """Default mapping has declared open conditions."""
        mapping = MeasurementMapping()
        open_c = mapping.open_conditions()
        assert len(open_c) > 0

    def test_open_conditions_do_not_block(self):
        """Open conditions do not make mapping inadmissible."""
        mapping = MeasurementMapping()
        assert mapping.admissible() is True

    def test_inadmissible_condition_blocks(self):
        """INADMISSIBLE condition makes mapping inadmissible."""
        mapping = MeasurementMapping()
        mapping.constraints.append(AdmissibilityCondition(
            name="test_failure",
            status=AdmissibilityStatus.INADMISSIBLE,
            declaration="Synthetic inadmissible condition for test."
        ))
        assert mapping.admissible() is False

    def test_known_open_condition_names(self):
        """Expected open conditions are present by name."""
        mapping = MeasurementMapping()
        open_names = {c.name for c in mapping.open_conditions()}
        assert "column_names" in open_names
        assert "cohort_adjacency" in open_names
        assert "benchmark_equivalence" in open_names

    def test_report_contains_source(self):
        """Report string contains declared source."""
        mapping = MeasurementMapping()
        report = mapping.report()
        assert "SOA" in report

    def test_report_contains_open_condition_label(self):
        """Report string flags open conditions."""
        mapping = MeasurementMapping()
        report = mapping.report()
        assert "OPEN_CONDITION" in report.upper()


# ---- SessionDeclaration tests ------------------------------------------

class TestSessionDeclaration:

    def test_verify_passes_by_default(self):
        """Default session declaration passes verification."""
        decl = SessionDeclaration()
        decl.verify()   # should not raise

    def test_verify_raises_on_inadmissible(self):
        """verify() raises ValueError if INADMISSIBLE condition exists."""
        decl = SessionDeclaration()
        decl.mapping.constraints.append(AdmissibilityCondition(
            name="synthetic_fail",
            status=AdmissibilityStatus.INADMISSIBLE,
            declaration="Synthetic test condition."
        ))
        with pytest.raises(ValueError):
            decl.verify()

    def test_relations_declared_not_empty(self):
        """At least one relation is declared for this session."""
        decl = SessionDeclaration()
        assert len(decl.relations_declared) > 0

    def test_ring_excluded(self):
        """Ring topology is explicitly excluded."""
        decl = SessionDeclaration()
        excluded_text = " ".join(decl.relations_excluded).upper()
        assert "RING" in excluded_text

    def test_cohort_excluded(self):
        """Cohort adjacency is explicitly excluded."""
        decl = SessionDeclaration()
        excluded_text = " ".join(decl.relations_excluded).upper()
        assert "COHORT" in excluded_text

    def test_discarded_not_empty(self):
        """Discarded information is explicitly declared — C constraint."""
        decl = SessionDeclaration()
        assert len(decl.discarded) > 0

    def test_detection_target_declared(self):
        """Detection target is declared before operators act."""
        decl = SessionDeclaration()
        assert len(decl.detection_target) > 0

    def test_report_contains_domain(self):
        """Report contains domain declaration."""
        decl = SessionDeclaration()
        report = decl.report()
        assert "Domain" in report

    def test_report_contains_discarded(self):
        """Report declares discarded information."""
        decl = SessionDeclaration()
        report = decl.report()
        assert "Discarded" in report

    def test_open_conditions_carried_through(self):
        """Open conditions from mapping are accessible via session."""
        decl = SessionDeclaration()
        open_c = decl.mapping.open_conditions()
        assert len(open_c) > 0
