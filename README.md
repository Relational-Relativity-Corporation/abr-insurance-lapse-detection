# ABR Insurance Lapse Detection

Relational observational layer applied to life insurance lapse experience data,
demonstrating early detection of shock lapse structure using the ABRCE invariant
relational kernel.

## What This Is

Standard lapse models treat each policy as an independent entity — scoring per-policy
attributes without declaring the relational structure between policyholders sharing
economic environments, product cohorts, or premium jump exposure.

This repo applies the ABRCE relational kernel as an observational layer on publicly
available SOA post-level term lapse experience data. The question is whether relational
gradients across the cell structure — premium jump magnitude, policy duration, issue
cohort — carry signal that per-cell index-local analysis may not surface.

This is a proof of concept, not a production system. Results are bounded by the
declared measurement mapping. No claim is made beyond the declared domain.

## Data Source

**SOA 2014 Post-Level Term Lapse and Mortality Study**
Society of Actuaries / RGA Reinsurance Company
https://www.soa.org/resources/experience-studies/2014/research-2014-post-level-shock/

Download: `research-2014-post-level-term-lapse-study.zip`
Place extracted files in `data/` before running experiments.

**Data is not committed to this repo.** It is publicly available directly from the SOA.
See `data/README.md` for download and placement instructions.

**Declared measurement mapping constraints:**
- Data is aggregated industry experience across multiple companies — not company-specific
- Each row represents a cell: a unique combination of study year, duration, gender,
  issue age, face amount, and premium jump magnitude
- Individual policy-level relational structure has been partially discarded through
  aggregation — this is declared, not assumed absent
- The cell structure itself carries partial relational information: cells adjacent in
  premium jump magnitude and duration are declared neighbors in this analysis

## Relational Structure Declared

Nodes: experience cells (unique combinations of study year, duration, issue age,
gender, face amount band, premium jump band)

Declared relations:
- Duration adjacency: cells in consecutive policy years within the same cohort
- Premium jump adjacency: cells in neighboring premium jump magnitude bands
- Issue cohort adjacency: cells from the same issue year across consecutive durations

These relations are declared from the observable structure of the data, not inferred.
Provenance: SOA cell definition and study design.

## Detection Pipeline

```
A(x)[e]   = x[s] - x[t]          # relational gradient across declared cell adjacency
B(g)[e]   = g[e] + Σ succ(e) g   # accumulation along declared duration path
R(g)[e]   = g[e] + ρ(fwd - bwd)  # antisymmetric circulation
C(x)[i]   = x[i] / (1 + |x[i]|) # bounded coherence, output in (-1, 1)

relational_spread   = ||A(Q)||² / n      # variance by construction
relational_momentum = mean(|C(R(A(W)))|) # bounded trend signal
```

Applied to the lapse rate field Q over the declared cell adjacency structure.

**On B:** B is included here — unlike the robotics detection repo — because the
insurance application operates over declared duration paths, not snapshots. Accumulation
along the duration path is the correct operator for detecting how lapse pressure
propagates forward through policy years.

## Benchmark

The SOA 2014 study identifies shock lapse at the end of the level term period as the
primary risk event. The relational signal is compared against the timing and magnitude
of shock lapse as reported in the SOA study — not against a simulated failure threshold.

## Kernel

The mathematical foundation is open source:
https://github.com/Relational-Relativity-Corporation/abr-kernel

## Dependencies

numpy, pandas, matplotlib, openpyxl

## Run

```
pip install -e .
python experiments/run_analysis.py
```

## Status

Proof of concept. Data loading and operator application are functional on the SOA
Excel format. Benchmark comparison against SOA shock lapse findings is in progress.

---

*Metatron Dynamics, Inc. Bounded over D. No claim beyond D.*
