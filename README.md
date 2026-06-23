# ABR Insurance Lapse — Relational Spread Analysis

Relational observational layer applied to life insurance lapse experience data,
characterizing the spread of lapse rates across declared cell adjacency as policies
approach the end of the level term period.

## What This Is

Standard lapse models treat each policy cell as an independent unit — computing a
lapse rate per cell and asking whether that rate is high or low. That is an
index-local question. It does not observe how lapse rates are distributed across
cells that are relationally adjacent in premium jump exposure and policy duration.

This repo applies the ABRCE relational kernel as an observational layer on publicly
available SOA post-level term lapse experience data. The primary observable is
relational spread across the declared cell structure — how fractured or uniform the
lapse rate field is at each duration step.

When spread is low, policyholders behave similarly regardless of premium jump
exposure. When spread is high, the field is fracturing — cells are diverging in
lapse behavior across their declared neighbors. That fracturing is structurally
observable in the relational field.

**Declared finding on synthetic data calibrated to SOA aggregates:**
Relational spread peaks at duration 10 — the end of the declared 10-year level term
period — at 31.6x above the pre-peak baseline mean. The field fractures sharply and
precisely at the declared event boundary.

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

Relations excluded:
- Issue cohort adjacency: open condition — excluded pending declaration
- Ring topology: proof construct only, never admissible in application

These relations are declared from the observable structure of the data, not inferred.
Provenance: SOA cell definition and study design.

## Observable and Finding

The primary observable is relational spread in the lapse experience field, computed
per duration step over declared cell adjacency.

Spread is observed and reported — not thresholded. Elevation ratio and peak location
are the declared outputs.

```
Spread(duration d) = relational_spread(cell_lapse[cells at duration d])
```

Relational momentum over the spread series is computed as a secondary characterization
of directional trend. It does not gate the primary finding.

**On B:** B accumulation is included because the insurance application operates over
declared duration paths, not snapshots. Accumulation along the duration path is the
correct operator for how lapse pressure propagates through policy years.

**On the observable:** Spread lives in the lapse experience field (cell_lapse).
The ABRCE operators construct relational context — ρ, edge structure, circulation.
The finding lives in the lapse field, not the kernel output.

## Kernel

The mathematical foundation is open source:
https://github.com/Relational-Relativity-Corporation/abr-kernel

## Dependencies

numpy, pandas, matplotlib, openpyxl

## Run

```
pip install -e .
python experiments/run_analysis.py --synthetic
python experiments/run_analysis.py --filepath data/SOA_lapse_study.xlsx
```

## Open Conditions

- **column_names:** SOA Excel column names must be verified against the actual
  downloaded file before running on real data
- **cohort_adjacency:** Issue cohort adjacency is structurally available but not
  declared for use — excluded pending declaration of what relational claim it makes
- **baseline_window:** Pre-peak baseline uses all durations before the spread peak;
  sensitivity to window choice not yet characterized

## Status

Proof of concept. Pipeline fully operational on synthetic data calibrated to SOA
aggregates. Spread peak confirmed at duration 10, 31.6x elevation above baseline.
Verification on actual SOA data pending column name resolution.

---

*Metatron Dynamics, Inc. Bounded over D. No claim beyond D.*
