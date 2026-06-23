# tests/test_operators.py — Invariant tests for ABRCE operators
#
# These are invariant tests, not behavioral tests.
# Each test verifies a declared mathematical property.
# If any test fails the operator violates its declared constraint.
#
# No SOA data required. All tests use declared synthetic fields.
# Bounded over D. No claim beyond D.

import pytest
import numpy as np
from abr_insurance.operators import (
    DeclaredRelations, EdgeType,
    operator_a, operator_b, operator_r, operator_c,
    compute_rho, relational_spread, relational_momentum
)


# ---- Fixtures ----------------------------------------------------------

def path_graph(n: int) -> DeclaredRelations:
    """
    Open path: 0→1→2→...→(n-1)
    All DURATION edges. No ring. Terminal has no successor.
    """
    edges = [(i, i + 1, EdgeType.DURATION) for i in range(n - 1)]
    return DeclaredRelations.from_typed_edges(n, edges)


def mixed_graph() -> DeclaredRelations:
    """
    4 nodes.
    DURATION: 0→1, 1→2
    JUMP:     0→3, 1→3
    """
    edges = [
        (0, 1, EdgeType.DURATION),
        (1, 2, EdgeType.DURATION),
        (0, 3, EdgeType.JUMP),
        (1, 3, EdgeType.JUMP),
    ]
    return DeclaredRelations.from_typed_edges(4, edges)


def fan_graph() -> DeclaredRelations:
    """
    Fan-out DAG: 0→1, 0→2, 1→3, 2→3
    All DURATION. Node 3 is terminal.
    """
    edges = [
        (0, 1, EdgeType.DURATION),
        (0, 2, EdgeType.DURATION),
        (1, 3, EdgeType.DURATION),
        (2, 3, EdgeType.DURATION),
    ]
    return DeclaredRelations.from_typed_edges(4, edges)


# ---- DeclaredRelations structural tests --------------------------------

class TestDeclaredRelations:

    def test_open_boundary_not_wrapped(self):
        """Terminal edge has no DURATION successor — not wrapped."""
        rel = path_graph(3)
        last_edge = rel.n_edges() - 1
        assert rel.succ_by_type(last_edge, EdgeType.DURATION) == []

    def test_terminal_node_no_outgoing(self):
        """Terminal node has no outgoing edges of any type."""
        rel = path_graph(4)
        assert rel.out[3] == []

    def test_fan_out_degree(self):
        """Fan-out node has correct out-degree."""
        rel = fan_graph()
        assert len(rel.out[0]) == 2

    def test_fan_in_degree(self):
        """Fan-in node has correct in-degree."""
        rel = fan_graph()
        assert len(rel.inc[3]) == 2

    def test_jump_edges_not_in_duration_successors(self):
        """JUMP edges must not appear in DURATION successor traversal."""
        rel = mixed_graph()
        for e in range(rel.n_edges()):
            if rel.edge_type(e) == EdgeType.DURATION:
                succs = rel.succ_by_type(e, EdgeType.DURATION)
                for f in succs:
                    assert rel.edge_type(f) == EdgeType.DURATION

    def test_edges_of_type_correct(self):
        """edges_of_type returns only edges of declared type."""
        rel = mixed_graph()
        dur_edges = rel.edges_of_type(EdgeType.DURATION)
        jmp_edges = rel.edges_of_type(EdgeType.JUMP)
        assert all(rel.edge_type(e) == EdgeType.DURATION for e in dur_edges)
        assert all(rel.edge_type(e) == EdgeType.JUMP for e in jmp_edges)
        assert len(dur_edges) + len(jmp_edges) == rel.n_edges()

    def test_valid_node_indices(self):
        """All edge endpoints reference valid node indices."""
        rel = mixed_graph()
        for s, t, _ in rel.edges:
            assert 0 <= s < rel.n_nodes
            assert 0 <= t < rel.n_nodes


# ---- Operator A tests --------------------------------------------------

class TestOperatorA:

    def test_zero_sum(self):
        """A(x) over a path graph is not necessarily zero-sum by node,
        but each edge is a declared directed difference."""
        rel = path_graph(4)
        f = np.array([1.0, 2.0, 4.0, 8.0])
        a = operator_a(f, rel)
        # Each edge: src - tgt
        assert a[0] == pytest.approx(f[0] - f[1])
        assert a[1] == pytest.approx(f[1] - f[2])
        assert a[2] == pytest.approx(f[2] - f[3])

    def test_identical_field_zero_gradient(self):
        """Identical values produce zero gradient on all edges."""
        rel = path_graph(5)
        f = np.ones(5) * 3.7
        a = operator_a(f, rel)
        assert np.allclose(a, 0.0)

    def test_direction_declared(self):
        """A(x)[e] = x[src] - x[tgt] — direction follows declaration."""
        rel = DeclaredRelations.from_typed_edges(
            2, [(0, 1, EdgeType.DURATION)]
        )
        f = np.array([5.0, 2.0])
        a = operator_a(f, rel)
        assert a[0] == pytest.approx(3.0)   # 5 - 2

    def test_reversed_direction(self):
        """Reversing edge direction reverses gradient sign."""
        rel_fwd = DeclaredRelations.from_typed_edges(
            2, [(0, 1, EdgeType.DURATION)]
        )
        rel_rev = DeclaredRelations.from_typed_edges(
            2, [(1, 0, EdgeType.DURATION)]
        )
        f = np.array([5.0, 2.0])
        assert operator_a(f, rel_fwd)[0] == pytest.approx(
            -operator_a(f, rel_rev)[0]
        )

    def test_output_shape(self):
        """Output shape matches number of declared edges."""
        rel = mixed_graph()
        f = np.array([1.0, 2.0, 3.0, 4.0])
        a = operator_a(f, rel)
        assert a.shape == (rel.n_edges(),)

    def test_domain_preserved(self):
        """Output values are finite for finite input."""
        rel = path_graph(6)
        f = np.array([0.01, 0.05, 0.12, 0.45, 0.80, 0.95])
        a = operator_a(f, rel)
        assert np.all(np.isfinite(a))


# ---- Operator B tests --------------------------------------------------

class TestOperatorB:

    def test_terminal_accumulates_nothing(self):
        """Terminal DURATION edge accumulates nothing beyond itself."""
        rel = path_graph(3)
        a = np.array([1.0, 2.0])
        b = operator_b(a, rel)
        # Last DURATION edge (index 1) has no successor
        assert b[1] == pytest.approx(a[1])

    def test_accumulates_forward(self):
        """Non-terminal edge accumulates successor value."""
        rel = path_graph(3)
        a = np.array([1.0, 2.0])
        b = operator_b(a, rel)
        # Edge 0 successor is edge 1
        assert b[0] == pytest.approx(a[0] + a[1])

    def test_jump_edges_not_accumulated(self):
        """JUMP edges do not participate in B accumulation."""
        rel = mixed_graph()
        f = np.array([0.1, 0.2, 0.3, 0.8])
        a = operator_a(f, rel)
        b = operator_b(a, rel)
        # JUMP edges: their values should equal a values (no accumulation)
        for e in rel.edges_of_type(EdgeType.JUMP):
            assert b[e] == pytest.approx(a[e])

    def test_zero_input_zero_output(self):
        """Zero gradient field produces zero accumulated field."""
        rel = path_graph(5)
        a = np.zeros(rel.n_edges())
        b = operator_b(a, rel)
        assert np.allclose(b, 0.0)

    def test_output_shape(self):
        """Output shape matches input shape."""
        rel = mixed_graph()
        a = np.ones(rel.n_edges())
        b = operator_b(a, rel)
        assert b.shape == a.shape


# ---- Operator R tests --------------------------------------------------

class TestOperatorR:

    def test_antisymmetry_uniform_rho(self):
        """R detects directional asymmetry — rising path has positive
        net forward pressure."""
        rel = path_graph(4)
        # Rising gradient: more forward than behind at each edge
        a = np.array([1.0, 2.0, 3.0])
        rho = np.ones(4) * 0.4
        b = operator_b(a, rel)
        r = operator_r(b, rel, rho)
        # First edge should have net positive forward pressure
        assert r[0] > b[0]

    def test_zero_rho_passthrough(self):
        """R with rho=0 passes b_field unchanged."""
        rel = path_graph(4)
        a = np.array([1.0, 2.0, 3.0])
        rho = np.zeros(4)
        b = operator_b(a, rel)
        r = operator_r(b, rel, rho)
        assert np.allclose(r, b)

    def test_jump_couples_within_type(self):
        """JUMP edges couple only within JUMP type in R."""
        rel = mixed_graph()
        f = np.array([0.1, 0.5, 0.2, 0.9])
        a = operator_a(f, rel)
        rho = compute_rho(a, rel, 0.4)
        b = operator_b(a, rel)
        r = operator_r(b, rel, rho)
        # Result should be finite for all edges
        assert np.all(np.isfinite(r))

    def test_output_shape(self):
        """Output shape matches input shape."""
        rel = path_graph(5)
        a = np.random.rand(rel.n_edges())
        rho = np.ones(rel.n_nodes) * 0.4
        b = operator_b(a, rel)
        r = operator_r(b, rel, rho)
        assert r.shape == a.shape


# ---- Operator C tests --------------------------------------------------

class TestOperatorC:

    def test_output_strictly_bounded(self):
        """C output is strictly in (-1, 1) for all finite nonzero input."""
        x = np.array([-100.0, -1.0, -0.1, 0.1, 1.0, 100.0])
        c = operator_c(x)
        assert np.all(c > -1.0)
        assert np.all(c < 1.0)

    def test_zero_maps_to_zero(self):
        """C(0) = 0."""
        assert operator_c(np.array([0.0]))[0] == pytest.approx(0.0)

    def test_sign_preserved(self):
        """C preserves sign of input."""
        x = np.array([-5.0, -0.1, 0.1, 5.0])
        c = operator_c(x)
        assert np.all(np.sign(c) == np.sign(x))

    def test_ordering_preserved(self):
        """C preserves ordering — monotone increasing."""
        x = np.array([0.1, 0.5, 1.0, 2.0, 10.0])
        c = operator_c(x)
        assert np.all(np.diff(c) > 0)

    def test_large_input_approaches_unity(self):
        """Large inputs approach ±1 asymptotically."""
        c_large = operator_c(np.array([1e6]))[0]
        assert abs(c_large) > 0.999

    def test_output_shape(self):
        """Output shape matches input shape."""
        x = np.random.rand(10)
        assert operator_c(x).shape == x.shape


# ---- ρ tests -----------------------------------------------------------

class TestRho:

    def test_bounded_by_rho_base(self):
        """ρ[i] ∈ [0, rho_base] for all nodes."""
        rel = path_graph(5)
        f = np.array([0.1, 0.5, 0.2, 0.8, 0.3])
        a = operator_a(f, rel)
        rho_base = 0.4
        rho = compute_rho(a, rel, rho_base)
        assert np.all(rho >= 0.0)
        assert np.all(rho <= rho_base + 1e-9)

    def test_zero_gradient_zero_rho(self):
        """Uniform field produces zero gradient, hence zero ρ."""
        rel = path_graph(5)
        f = np.ones(5) * 2.5
        a = operator_a(f, rel)
        rho = compute_rho(a, rel, 0.4)
        assert np.allclose(rho, 0.0)

    def test_local_not_global(self):
        """ρ at isolated node is not influenced by distant gradients."""
        # Node 0 connects only to node 1.
        # Node 3 connects only to node 4.
        # Large gradient at 3→4 should not affect ρ[0].
        rel = DeclaredRelations.from_typed_edges(5, [
            (0, 1, EdgeType.DURATION),
            (3, 4, EdgeType.DURATION),
        ])
        f = np.array([0.1, 0.1, 0.0, 0.1, 0.9])
        a = operator_a(f, rel)
        rho = compute_rho(a, rel, 0.4)
        # Node 0: gradient at edge 0→1 is tiny
        assert rho[0] < 0.01
        # Node 3: gradient at edge 3→4 is large
        assert rho[3] > 0.1

    def test_output_shape(self):
        """ρ shape matches number of nodes."""
        rel = path_graph(6)
        a = np.random.rand(rel.n_edges())
        rho = compute_rho(a, rel, 0.4)
        assert rho.shape == (rel.n_nodes,)


# ---- relational_spread tests -------------------------------------------

class TestRelationalSpread:

    def test_equals_population_variance(self):
        """relational_spread equals np.var (population, ddof=0)."""
        f = np.array([0.1, 0.3, 0.5, 0.2, 0.4])
        assert relational_spread(f) == pytest.approx(np.var(f))

    def test_uniform_field_zero_spread(self):
        """Uniform field has zero spread."""
        f = np.ones(6) * 0.25
        assert relational_spread(f) == pytest.approx(0.0)

    def test_nonnegative(self):
        """Spread is always non-negative."""
        for _ in range(10):
            f = np.random.rand(8)
            assert relational_spread(f) >= 0.0

    def test_scale_sensitivity(self):
        """Spread scales with square of field magnitude."""
        f = np.array([0.1, 0.3, 0.5])
        assert relational_spread(2 * f) == pytest.approx(4 * relational_spread(f))


# ---- relational_momentum tests -----------------------------------------

class TestRelationalMomentum:

    def test_output_length_matches_input(self):
        """Output length matches input series length."""
        series = np.random.rand(20)
        mom = relational_momentum(series, window=5)
        assert len(mom) == len(series)

    def test_nan_before_window(self):
        """Values before window are NaN."""
        series = np.random.rand(15)
        window = 5
        mom = relational_momentum(series, window=window)
        assert np.all(np.isnan(mom[:window - 1]))

    def test_nonnegative_after_window(self):
        """Momentum values after window fill are non-negative."""
        series = np.linspace(0.01, 0.5, 20)
        mom = relational_momentum(series, window=5)
        valid = mom[~np.isnan(mom)]
        assert np.all(valid >= 0.0)

    def test_bounded_by_unity(self):
        """Momentum output bounded in [0, 1) — inherits C bound."""
        series = np.random.rand(20) * 100.0
        mom = relational_momentum(series, window=5)
        valid = mom[~np.isnan(mom)]
        assert np.all(valid < 1.0)
        assert np.all(valid >= 0.0)

    def test_rising_series_produces_signal(self):
        """Strongly rising series produces momentum above zero."""
        series = np.linspace(0.0, 1.0, 30)
        mom = relational_momentum(series, window=5)
        valid = mom[~np.isnan(mom)]
        assert np.any(valid > 0.01)

    def test_flat_series_low_signal(self):
        """Flat series produces near-zero momentum."""
        series = np.ones(20) * 0.3
        mom = relational_momentum(series, window=5)
        valid = mom[~np.isnan(mom)]
        assert np.all(valid < 0.01)
