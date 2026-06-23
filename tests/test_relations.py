# tests/test_relations.py — Structural correctness of declared edges
#
# Tests verify that declared relational structure over SOA cell data:
# - produces correct edge types
# - respects open boundaries
# - excludes cohort adjacency by default
# - contains no fabricated edges
# - correctly separates DURATION from JUMP participation in B
#
# All tests use synthetic cell DataFrames — no SOA file required.
# Bounded over D. No claim beyond D.

import pytest
import pandas as pd
import numpy as np
from abr_insurance.operators import EdgeType
from abr_insurance.relations import (
    build_cell_index,
    duration_edges,
    premium_jump_edges,
    cohort_edges,
    build_declared_relations,
)


# ---- Fixtures ----------------------------------------------------------

def simple_cells() -> pd.DataFrame:
    """
    Minimal synthetic cell structure.
    2 issue years, 3 durations, 2 jump bands.
    24 cells total.
    """
    rows = []
    for year in [2000, 2001]:
        for dur in [1, 2, 3]:
            for jump in [1, 2]:
                rows.append({
                    'IssueYear': year,
                    'Duration': dur,
                    'PremiumJumpBand': jump,
                    'LapseRate': np.random.uniform(0.01, 0.5)
                })
    return pd.DataFrame(rows)


def single_cohort_cells() -> pd.DataFrame:
    """Single issue year, 4 durations, 1 jump band — pure path."""
    rows = [
        {'IssueYear': 2000, 'Duration': d,
         'PremiumJumpBand': 1, 'LapseRate': 0.1 * d}
        for d in [1, 2, 3, 4]
    ]
    return pd.DataFrame(rows)


# ---- build_cell_index tests --------------------------------------------

class TestBuildCellIndex:

    def test_unique_cell_indices(self):
        """Every cell gets a unique integer index."""
        df = simple_cells()
        df_indexed, cells = build_cell_index(df)
        assert cells['cell_idx'].nunique() == len(cells)

    def test_cell_idx_contiguous(self):
        """Cell indices are contiguous from 0."""
        df = simple_cells()
        _, cells = build_cell_index(df)
        assert sorted(cells['cell_idx'].tolist()) == list(range(len(cells)))

    def test_cell_idx_in_df(self):
        """cell_idx column added to returned dataframe."""
        df = simple_cells()
        df_indexed, _ = build_cell_index(df)
        assert 'cell_idx' in df_indexed.columns

    def test_no_missing_cell_idx(self):
        """No NaN cell indices in returned dataframe."""
        df = simple_cells()
        df_indexed, _ = build_cell_index(df)
        assert df_indexed['cell_idx'].notna().all()


# ---- duration_edges tests ----------------------------------------------

class TestDurationEdges:

    def test_direction_forward_in_time(self):
        """Duration edges point from earlier to later duration."""
        df = single_cohort_cells()
        _, cells = build_cell_index(df)
        edges = duration_edges(cells)
        for s, t in edges:
            src_dur = cells.loc[cells['cell_idx'] == s, 'Duration'].values[0]
            tgt_dur = cells.loc[cells['cell_idx'] == t, 'Duration'].values[0]
            assert src_dur < tgt_dur

    def test_no_wraparound(self):
        """Last duration cell has no outgoing duration edge."""
        df = single_cohort_cells()
        _, cells = build_cell_index(df)
        edges = duration_edges(cells)
        max_dur = cells['Duration'].max()
        terminal_idx = cells.loc[
            cells['Duration'] == max_dur, 'cell_idx'
        ].values[0]
        sources = [s for s, _ in edges]
        assert terminal_idx not in sources

    def test_count_correct(self):
        """Edge count = (n_durations - 1) * n_jump_bands * n_cohorts."""
        df = simple_cells()
        _, cells = build_cell_index(df)
        edges = duration_edges(cells)
        n_durations = cells['Duration'].nunique()
        n_jumps = cells['PremiumJumpBand'].nunique()
        n_cohorts = cells['IssueYear'].nunique()
        expected = (n_durations - 1) * n_jumps * n_cohorts
        assert len(edges) == expected

    def test_no_self_loops(self):
        """No edge from a cell to itself."""
        df = simple_cells()
        _, cells = build_cell_index(df)
        edges = duration_edges(cells)
        assert all(s != t for s, t in edges)

    def test_valid_node_indices(self):
        """All edge endpoints are valid cell indices."""
        df = simple_cells()
        _, cells = build_cell_index(df)
        edges = duration_edges(cells)
        n_nodes = len(cells)
        assert all(0 <= s < n_nodes and 0 <= t < n_nodes for s, t in edges)


# ---- premium_jump_edges tests ------------------------------------------

class TestPremiumJumpEdges:

    def test_direction_increasing_jump(self):
        """Jump edges point from lower to higher jump band."""
        df = simple_cells()
        _, cells = build_cell_index(df)
        edges = premium_jump_edges(cells)
        for s, t in edges:
            src_jump = cells.loc[cells['cell_idx'] == s, 'PremiumJumpBand'].values[0]
            tgt_jump = cells.loc[cells['cell_idx'] == t, 'PremiumJumpBand'].values[0]
            assert src_jump < tgt_jump

    def test_count_correct(self):
        """Edge count = (n_jumps - 1) * n_durations * n_cohorts."""
        df = simple_cells()
        _, cells = build_cell_index(df)
        edges = premium_jump_edges(cells)
        n_durations = cells['Duration'].nunique()
        n_jumps = cells['PremiumJumpBand'].nunique()
        n_cohorts = cells['IssueYear'].nunique()
        expected = (n_jumps - 1) * n_durations * n_cohorts
        assert len(edges) == expected

    def test_no_self_loops(self):
        """No edge from a cell to itself."""
        df = simple_cells()
        _, cells = build_cell_index(df)
        edges = premium_jump_edges(cells)
        assert all(s != t for s, t in edges)

    def test_valid_node_indices(self):
        """All edge endpoints are valid cell indices."""
        df = simple_cells()
        _, cells = build_cell_index(df)
        edges = premium_jump_edges(cells)
        n_nodes = len(cells)
        assert all(0 <= s < n_nodes and 0 <= t < n_nodes for s, t in edges)


# ---- build_declared_relations tests ------------------------------------

class TestBuildDeclaredRelations:

    def test_cohort_excluded_by_default(self):
        """COHORT edges absent when include_cohort=False."""
        df = simple_cells()
        _, cells = build_cell_index(df)
        rel = build_declared_relations(cells)
        cohort = rel.edges_of_type(EdgeType.COHORT)
        assert len(cohort) == 0

    def test_cohort_included_when_declared(self):
        """COHORT edges present when include_cohort=True."""
        df = simple_cells()
        _, cells = build_cell_index(df)
        rel = build_declared_relations(cells, include_cohort=True)
        cohort = rel.edges_of_type(EdgeType.COHORT)
        assert len(cohort) > 0

    def test_duration_edges_typed_correctly(self):
        """All DURATION edges carry DURATION type."""
        df = simple_cells()
        _, cells = build_cell_index(df)
        rel = build_declared_relations(cells)
        for e in rel.edges_of_type(EdgeType.DURATION):
            assert rel.edge_type(e) == EdgeType.DURATION

    def test_jump_edges_typed_correctly(self):
        """All JUMP edges carry JUMP type."""
        df = simple_cells()
        _, cells = build_cell_index(df)
        rel = build_declared_relations(cells)
        for e in rel.edges_of_type(EdgeType.JUMP):
            assert rel.edge_type(e) == EdgeType.JUMP

    def test_jump_not_in_duration_successors(self):
        """JUMP edges never appear in DURATION successor traversal."""
        df = simple_cells()
        _, cells = build_cell_index(df)
        rel = build_declared_relations(cells)
        for e in rel.edges_of_type(EdgeType.DURATION):
            succs = rel.succ_by_type(e, EdgeType.DURATION)
            for f in succs:
                assert rel.edge_type(f) == EdgeType.DURATION

    def test_no_fabricated_edges(self):
        """All edge endpoints are valid node indices."""
        df = simple_cells()
        _, cells = build_cell_index(df)
        rel = build_declared_relations(cells, include_cohort=True)
        for s, t, _ in rel.edges:
            assert 0 <= s < rel.n_nodes
            assert 0 <= t < rel.n_nodes

    def test_n_nodes_matches_cells(self):
        """n_nodes matches number of declared cells."""
        df = simple_cells()
        _, cells = build_cell_index(df)
        rel = build_declared_relations(cells)
        assert rel.n_nodes == len(cells)

    def test_duration_only_has_no_jump(self):
        """include_jump=False produces no JUMP edges."""
        df = simple_cells()
        _, cells = build_cell_index(df)
        rel = build_declared_relations(cells, include_jump=False)
        assert len(rel.edges_of_type(EdgeType.JUMP)) == 0

    def test_no_ring(self):
        """No edge loops back from terminal to initial node."""
        df = single_cohort_cells()
        _, cells = build_cell_index(df)
        rel = build_declared_relations(cells)
        max_dur = cells['Duration'].max()
        min_dur = cells['Duration'].min()
        terminal_idx = cells.loc[
            cells['Duration'] == max_dur, 'cell_idx'
        ].values[0]
        initial_idx = cells.loc[
            cells['Duration'] == min_dur, 'cell_idx'
        ].values[0]
        edge_pairs = [(s, t) for s, t, _ in rel.edges]
        assert (terminal_idx, initial_idx) not in edge_pairs
