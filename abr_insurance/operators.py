# operators.py — ABR Insurance Lapse Detection
# ABRCE invariant relational kernel applied to insurance lapse experience field.
#
# Kernel: E(x, ρ) = R(B(A(x)), ρ(A(x)))    [A → B → R → E]
#
# Edge typing — declared:
#   DURATION edges carry B accumulation and R circulation.
#              Temporal continuation along policy year path.
#   JUMP edges carry R cross-sectional coupling only.
#              Premium jump is a cell characteristic at end-of-term
#              shock point, not a temporal continuation relation.
#              Symetra term products carry level premiums throughout
#              the initial term; the jump is a single end-of-term event.
#   COHORT edges — open condition, excluded by default.
#
# C is a declared projection, not a kernel operator.
# No ring. No default topology. No relation fabricated.
# Bounded over D. No claim beyond D.

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple


# ---- Edge type declaration ---------------------------------------------

class EdgeType(Enum):
    DURATION = "duration"   # temporal continuation — B and R
    JUMP     = "jump"       # cross-sectional shock — R only
    COHORT   = "cohort"     # open condition — excluded by default


# ---- Declared relational structure -------------------------------------

@dataclass
class DeclaredRelations:
    """
    Typed cell adjacency for the SOA lapse experience field.

    Nodes: experience cells
    Edges: typed directed relations with declared provenance
    """
    n_nodes: int
    edges: List[Tuple[int, int, EdgeType]]  # (src, tgt, type)
    out: List[List[int]]                    # out[i]: edge indices leaving node i
    inc: List[List[int]]                    # inc[i]: edge indices entering node i

    @classmethod
    def from_typed_edges(cls, n_nodes: int,
                         typed_edges: List[Tuple[int, int, EdgeType]]):
        assert all(s < n_nodes and t < n_nodes for s, t, _ in typed_edges)
        out = [[] for _ in range(n_nodes)]
        inc = [[] for _ in range(n_nodes)]
        for e, (s, t, _) in enumerate(typed_edges):
            out[s].append(e)
            inc[t].append(e)
        return cls(n_nodes=n_nodes, edges=typed_edges, out=out, inc=inc)

    def n_edges(self): return len(self.edges)

    def edge_type(self, e: int) -> EdgeType:
        return self.edges[e][2]

    def succ_by_type(self, e: int, etype: EdgeType) -> List[int]:
        """Successor edges of declared type only."""
        tgt = self.edges[e][1]
        return [f for f in self.out[tgt] if self.edges[f][2] == etype]

    def pred_by_type(self, e: int, etype: EdgeType) -> List[int]:
        """Predecessor edges of declared type only."""
        src = self.edges[e][0]
        return [f for f in self.inc[src] if self.edges[f][2] == etype]

    def edges_of_type(self, etype: EdgeType) -> List[int]:
        """All edge indices of declared type."""
        return [i for i, (_, _, t) in enumerate(self.edges) if t == etype]


# ---- A — relational gradient extraction --------------------------------

def operator_a(field: np.ndarray, rel: DeclaredRelations) -> np.ndarray:
    """
    A(x)[e] = x[s] - x[t] over each declared edge e = (s, t).

    Applied uniformly across all declared edge types — gradient
    extraction does not depend on edge type.

    field   : (n_nodes,) lapse rate field over experience cells
    returns : (n_edges,) gradient field
    """
    assert field.shape == (rel.n_nodes,)
    return np.array([field[s] - field[t] for s, t, _ in rel.edges])


# ---- ρ — local contrast ------------------------------------------------

def compute_rho(a_field: np.ndarray, rel: DeclaredRelations,
                rho_base: float = 0.4) -> np.ndarray:
    """
    ρ[i] = ρ_base * m[i] / (1 + m[i])

    Local contrast derived per node from A(x).
    All incident edges contribute regardless of type —
    ρ reflects total local gradient magnitude.
    No aggregation beyond the node.

    returns: (n_nodes,) contrast field
    """
    rho = np.zeros(rel.n_nodes)
    for i in range(rel.n_nodes):
        incident = rel.out[i] + rel.inc[i]
        if incident:
            m = max(abs(a_field[e]) for e in incident)
            rho[i] = rho_base * m / (1.0 + m)
    return rho


# ---- B — accumulation along DURATION edges only ------------------------

def operator_b(a_field: np.ndarray, rel: DeclaredRelations) -> np.ndarray:
    """
    B(g)[e] = g[e] + Σ_{f ∈ succ_DURATION(e)} g[f]

    Accumulates along declared DURATION continuation only.
    JUMP and COHORT edges do not participate — premium jump is a
    cross-sectional characteristic at end-of-term, not a temporal path.
    Terminal duration edges accumulate nothing. No boundary closed.

    returns: (n_edges,) accumulated gradient field
    """
    b = a_field.copy()
    for e in range(rel.n_edges()):
        if rel.edge_type(e) == EdgeType.DURATION:
            for f in rel.succ_by_type(e, EdgeType.DURATION):
                b[e] += a_field[f]
    return b


# ---- R — antisymmetric circulation -------------------------------------

def operator_r(b_field: np.ndarray, rel: DeclaredRelations,
               rho: np.ndarray) -> np.ndarray:
    """
    R — typed antisymmetric circulation.

    For DURATION edges:
        R(g)[e] = g[e] + ρ[src] * (Σ succ_DURATION - Σ pred_DURATION)
        Detects directional asymmetry along temporal path.

    For JUMP edges:
        R(g)[e] = g[e] + ρ[src] * (Σ succ_JUMP - Σ pred_JUMP)
        Cross-sectional asymmetry across shock magnitude bands.
        Each relation type couples within its own declared class.

    returns: (n_edges,) circulation field
    """
    r = b_field.copy()
    for e, (s, _, etype) in enumerate(rel.edges):
        fwd = sum(b_field[f] for f in rel.succ_by_type(e, etype))
        bwd = sum(b_field[p] for p in rel.pred_by_type(e, etype))
        r[e] += rho[s] * (fwd - bwd)
    return r


# ---- C — declared projection -------------------------------------------

def operator_c(field: np.ndarray) -> np.ndarray:
    """
    C(x)[i] = x[i] / (1 + |x[i]|)

    Bounded coherence. Output in (-1, 1) by construction.
    Declared projection — preserves sign and ordering,
    discards magnitude beyond unit scale.
    """
    return field / (1.0 + np.abs(field))


# ---- E — kernel composition --------------------------------------------

def operator_e(field: np.ndarray, rel: DeclaredRelations,
               rho_base: float = 0.4) -> np.ndarray:
    """
    E(x, ρ) = R(B(A(x)), ρ(A(x)))

    Full kernel composition over declared typed relations.

    field   : (n_nodes,) lapse rate field over experience cells
    returns : (n_edges,) relational field
    """
    a = operator_a(field, rel)
    rho = compute_rho(a, rel, rho_base)
    b = operator_b(a, rel)
    return operator_r(b, rel, rho)


# ---- Detection functions -----------------------------------------------

def relational_spread(field: np.ndarray) -> float:
    """
    ||A(Q)||² / n — population variance by construction.

    Applied to the lapse rate field over all cells at a given
    duration or study year slice.
    """
    a = field - field.mean()
    return float(np.sum(a * a) / len(a))


def relational_momentum(series: np.ndarray, window: int = 10,
                        rho_base: float = 0.4) -> np.ndarray:
    """
    mean(|C(R(A(W)))|) over a rolling window of the spread series.

    Applied to the spread time series (1D) over a simple temporal chain.
    rel parameter removed — this function operates on the 1D spread
    series with implicit linear path, not the full SOA cell topology.
    B omitted at this layer — accumulation already captured in operator_e.

    returns: (n,) bounded trend signal
    """
    n = len(series)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        seg = series[i - window + 1: i + 1].tolist()
        mean = sum(seg) / len(seg)
        a = [x - mean for x in seg]
        r = [a[j] + rho_base * (
            (a[j + 1] if j + 1 < len(a) else 0.0) -
            (a[j - 1] if j - 1 >= 0 else 0.0)
        ) for j in range(len(a))]
        c = [x / (1.0 + abs(x)) for x in r]
        out[i] = float(np.mean(np.abs(c)))
    return out
