# pu_ratchet_analysis

Analysis code for procedural-unit (PU) acquisition across 3.3 million years of the hominin lithic
record, using the Paige & Perreault (2023) procedural unit dataset.

It accompanies an **earlier version** of the analysis in:

> D. Foley, "Technology as Boundary-Extending Motion: A Physical Mechanism for Hominin Evolution."
> Manuscript.

The current manuscript restricts some tests to the 145 hominin assemblages and adds further tests
that are not in this script. This script uses all 155 entries in the dataset. Code for the current
manuscript will be released separately.

## What it computes

1. Cumulative PU acquisition over time: linear vs exponential fit, with doubling time
2. Permutation test (10,000 permutations, seed 42) on the innovation rate of the last 500 kyr vs the first 2.8 Myr
3. PU count per assemblage by period
4. Irreversibility: whether any PU disappears after its first appearance

## Reproduce

```bash
git clone https://github.com/kodai-bot/pu_ratchet_analysis
cd pu_ratchet_analysis
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# Dataset: Paige & Perreault (2023), doi:10.5281/zenodo.7847839, v1.0.0
curl -L -o dataset.tar.gz \
  https://zenodo.org/api/records/7847839/files/Procedural.unit.dataset-1.0.tar.gz/content
tar xzf dataset.tar.gz

.venv/bin/python scripts/supplementary_pu_acceleration_analysis.py \
  "Procedural.unit.dataset-1.0/Dataset/procedural units obfus.csv"
```

Output: the statistics on the console, and `fig_pu_acceleration.pdf` (two panels).

The console output should match [`expected/output.txt`](expected/output.txt) exactly. For example:

```
Linear fit:               R² = 0.873, RSS = 315.50
Exponential fit:          R² = 0.955, RSS = 112.82
Doubling time:            929 kyr
p-value:                   0.0020   (10,000 permutations)
PUs lost after appearance:   0
```

The GitHub Actions workflow in `.github/workflows/reproduce.yml` downloads the dataset, checks its
SHA-256, runs the script and compares the output with `expected/output.txt` on every push.

Last verified: Python 3.14.4 with the versions pinned in `requirements.txt`.

## Data

Paige, J. and Perreault, C. (2023). *Dataset describing presence or absence of stone tool making
techniques in assemblages spanning the past 3 million years* (v1.0.0). Zenodo.
https://doi.org/10.5281/zenodo.7847839

The dataset is not redistributed here.

## Licence

Code: MIT (see `LICENSE`). The dataset has its own licence; see the Zenodo record.
