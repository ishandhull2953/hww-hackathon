# minihep

Foundation-model experiments on ATLAS Open Data `WW` events. Inspect with
`python minihep/inspect_branches.py`, load with `minihep.data`.

> This README is the canonical record of what exists here and is updated with
> every step. Current state: **step 1 done** (inspection + `data.load_events`),
> **step 2 done** (`prepare` split + scaler, tested). Model code does not exist yet.

## Data source

`../dataWW_d1.root` (relative to this directory), tree `tree_event`, 600,000
entries, 30 branches. 596,571 entries have `lep_n == 2`. Every event stores
exactly **two leptons** and **two jet slots** (some jets are absent even though
`jet_n` can exceed 2 — only the stored slots are usable).

## Branches used (all verified against the file)

| group | branches | notes |
|---|---|---|
| leptons | `lep_pt_{0,1}`, `lep_eta_{0,1}`, `lep_phi_{0,1}`, `lep_E_{0,1}` | pT in **GeV**, E in **MeV** (÷1000) |
| lepton ids | `lep_type_{0,1}` (PDG 11=e, 13=mu), `lep_charge_{0,1}` (±1) | |
| jets | `jet_pt_{0,1}`, `jet_eta_{0,1}`, `jet_phi_{0,1}`, `jet_E_{0,1}` | absent jet slot = **-7.0** filler; pad with zeros instead |
| MET | `met_et`, `met_phi` | GeV / rad |
| identity | `index`, `eventNumber`, `runNumber`, `channelNumber` | `index` = row in file |
| labels / weights | `label` (1=WW signal, 0=background), `mcWeight` | **not model inputs** |

The file has no jet flavour/type branch, so every jet is one token type.

## Token scheme

Slot order `[lep0, lep1, jet0, jet1, met]`; `(events, 5, 4)` in GeV.
`px = pT cos(phi)`, `py = pT sin(phi)`, `pz = pT sinh(eta)`, `E` from the
file (÷1000). MET token is the transverse proxy
`(MET cos(phi), MET sin(phi), 0, MET)` — **not** a neutrino four-vector.

| code | token | meaning |
|---|---|---|
| 0 | e- | PDG 11, charge −1 |
| 1 | e+ | PDG 11, charge +1 |
| 2 | mu- | PDG 13, charge −1 |
| 3 | mu+ | PDG 13, charge +1 |
| 4 | jet | any jet |
| 5 | met | MET proxy |

## Usage

```python
from minihep import data as d

ev = d.load_events(max_events=6000, seed=42)  # deterministic
ev.fourvec      # (6000, 5, 4) float32 GeV
ev.token_types  # (6000, 5) int64 codes
ev.valid        # (6000, 5) bool, False = padded token
ev.label        # (6000,) int64 — target only, never a model input
ev.mc_weight    # (6000,) float64 — event weight, never a model input
print(d.format_event(ev, 0))
```

Notes:

- `load_events` subsamples `lep_n == 2` entries with `np.random.default_rng(seed).permutation`; output is deterministic for fixed (file, `max_events`, `seed`).
- No `mcWeight` filtering (the course notebook filters `mcWeight > 0`; here raw weights are returned).
- Only numpy/`uproot` are used — no JAX/ROOT dependencies in the loader.

## Split + scaling (`minihep.prepare`)

```python
from minihep import data as d
from minihep import prepare as p

ev = d.load_events()                 # 6000 events
prepared = p.prepare(ev)             # 70/15/15 by event_index
prepared.train / prepared.val / prepared.test   # EventBatch with SCALED fourvec
prepared.scaler                      # fitted stats
print(p.summarize(prepared))         # sizes + no-overlap check
```

- **Split**: stable splitmix64 hash of `event_index` (file row) mod 100 →
  `<70` train, `70–84` val, `85–99` test. Because it is keyed on the file row,
  the same event always lands in the same split for any `max_events`/`seed`
  (verified by test). Never use Python's `hash()` (process-salted, not
  reproducible). File order is already interleaved by channel/label, but the
  hash split makes no ordering assumption.
- **Scaling** (`FeatureScaler`): `asinh(x / 50 GeV)` then per-component
  standardization, fitted on **valid tokens of training events only**
  (`event_mask` restricts events, used by the tests). All splits are
  transformed with training stats; padded tokens are reset to exactly zero
  after transform (invalid tokens would otherwise become `(0-mean)/std`).
- Metadata (`label`, `mc_weight`, ids) is carried alongside but is never a
  token feature; `EventBatch.__post_init__` validates shapes.
- Run tests: `python -m unittest minihep.test_prepare -v` (stdlib unittest;
  no pytest). Covers split disjointness/sizes, split stability across sample
  sizes, manual-formula scaler check, and that changing validation data does
  not change the fitted scaler.
