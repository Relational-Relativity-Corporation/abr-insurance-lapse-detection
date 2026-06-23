# relations.py — Declared relational structure over SOA cell structure
#
# Declares adjacency between experience cells based on:
#   1. Duration adjacency — consecutive policy years within same cohort
#      Provenance: temporal continuation along policy year path.
#      Participates in B accumulation and R circulation.
#
#   2. Premium jump adjacency — neighboring premium jump magnitude bands
#      at the end-of-term shock point only.
#      Provenance: SOA study design treats jump bands as ordinal categories;
#      adjacency follows declared neighboring bands only.
#      Does NOT participate in B accumulation — premium jump is a cell
#      characteristic at a single point in time (end of level term), not
#      a temporal continuation relation. Participates in R cross-sectional
#      coupling only.
#
#   3. Issue cohort adjacency — same issue year across consecutive durations
#      Declared open condition: requires additional declaration of what
#      constitutes meaningful cohort coupling. Excluded by default.
#
# Note on Symetra product structure: Symetra term products carry level
# premiums throughout the initial term period. The premium jump is a
# single event at end of level term — not a mid-policy phenomenon.
# Premium jump bands therefore classify cells by shock magnitude at
# term boundary, not by any temporal sequence within the policy.
#
# No relation fabricated. Every edge has declared source.
# Bounded over D. No claim beyond D.

import pandas as pd
from typing import List, Tuple
from .operators import DeclaredRelations, EdgeType


# ---- Cell index construction -------------------------------------------

def build_cell_index(df: pd.DataFrame,
                     cohort_col: str = 'IssueYear',
                     duration_col: str = 'Duration',
                     jump_col: str = 'PremiumJumpBand') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Assign a unique integer index to each experience cell.
    Returns (df with cell_idx, cells DataFrame).
    """
    cell_keys = [cohort_col, duration_col, jump_col]
    df = df.copy()
    cells = df[cell_keys].drop_duplicates().reset_index(drop=True)
    cells['cell_idx'] = cells.index
    df = df.merge(cells, on=cell_keys, how='left')
    return df, cells


# ---- Declared relation builders ----------------------------------------

def duration_edges(cells: pd.DataFrame,
                   cohort_col: str = 'IssueYear',
                   duration_col: str = 'Duration',
                   jump_col: str = 'PremiumJumpBand') -> List[Tuple[int, int]]:
    """
    Duration adjacency: cell at duration t → cell at duration t+1
    within the same issue cohort and premium jump band.

    Provenance: declared temporal continuation along policy year path.
    Direction: earlier duration → later duration.
    Terminal cells have no successor — not wrapped.

    Participates in B accumulation and R circulation.
    EdgeType: DURATION
    """
    edges = []
    for _, grp in cells.groupby([cohort_col, jump_col]):
        grp_sorted = grp.sort_values(duration_col)
        idxs = grp_sorted['cell_idx'].tolist()
        for i in range(len(idxs) - 1):
            edges.append((idxs[i], idxs[i + 1]))
    return edges


def premium_jump_edges(cells: pd.DataFrame,
                       cohort_col: str = 'IssueYear',
                       duration_col: str = 'Duration',
                       jump_col: str = 'PremiumJumpBand') -> List[Tuple[int, int]]:
    """
    Premium jump adjacency: cell in jump band j → cell in jump band j+1
    within the same issue cohort and duration.

    Provenance: SOA study design treats jump bands as ordinal categories;
    adjacency follows declared neighboring bands only.

    Cross-sectional relation at end-of-term shock point only.
    Does NOT participate in B accumulation.
    Participates in R cross-sectional coupling only.
    EdgeType: JUMP
    """
    edges = []
    for _, grp in cells.groupby([cohort_col, duration_col]):
        grp_sorted = grp.sort_values(jump_col)
        idxs = grp_sorted['cell_idx'].tolist()
        for i in range(len(idxs) - 1):
            edges.append((idxs[i], idxs[i + 1]))
    return edges


def cohort_edges(cells: pd.DataFrame,
                 cohort_col: str = 'IssueYear',
                 duration_col: str = 'Duration',
                 jump_col: str = 'PremiumJumpBand') -> List[Tuple[int, int]]:
    """
    Issue cohort adjacency: cell in cohort y → cell in cohort y+1
    at the same duration and premium jump band.

    Open condition: requires explicit declaration of what constitutes
    meaningful cohort coupling before use. Excluded by default.
    EdgeType: COHORT
    """
    edges = []
    for _, grp in cells.groupby([duration_col, jump_col]):
        grp_sorted = grp.sort_values(cohort_col)
        idxs = grp_sorted['cell_idx'].tolist()
        for i in range(len(idxs) - 1):
            edges.append((idxs[i], idxs[i + 1]))
    return edges


# ---- Full declared relation bundle -------------------------------------

def build_declared_relations(cells: pd.DataFrame,
                              include_duration: bool = True,
                              include_jump: bool = True,
                              include_cohort: bool = False,
                              cohort_col: str = 'IssueYear',
                              duration_col: str = 'Duration',
                              jump_col: str = 'PremiumJumpBand') -> DeclaredRelations:
    """
    Build the full declared relational structure over SOA experience cells
    with typed edges.

    Edge types:
      DURATION — temporal continuation, participates in B and R
      JUMP     — cross-sectional shock comparison, participates in R only
      COHORT   — open condition, excluded by default

    Parameters
    ----------
    include_duration : declare duration adjacency (default True)
    include_jump     : declare premium jump adjacency (default True)
    include_cohort   : declare cohort adjacency (default False — open condition)

    Returns
    -------
    DeclaredRelations with typed edges.
    """
    n_nodes = len(cells)
    typed_edges = []

    if include_duration:
        for e in duration_edges(cells, cohort_col, duration_col, jump_col):
            typed_edges.append((e[0], e[1], EdgeType.DURATION))

    if include_jump:
        for e in premium_jump_edges(cells, cohort_col, duration_col, jump_col):
            typed_edges.append((e[0], e[1], EdgeType.JUMP))

    if include_cohort:
        for e in cohort_edges(cells, cohort_col, duration_col, jump_col):
            typed_edges.append((e[0], e[1], EdgeType.COHORT))

    # Deduplicate on (src, tgt, type)
    typed_edges = list(set(typed_edges))

    return DeclaredRelations.from_typed_edges(n_nodes, typed_edges)
