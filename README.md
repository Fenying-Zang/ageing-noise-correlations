# Ageing noise correlations

This repository contains the analysis code for the noise correlation paper:
*Age-related changes in noise correlations in a decision-making task*

The analysis builds on [Fano factor analysis](https://github.com/Fenying-Zang/Ageing_behavioral_and_neural_variability), but is organized as a separate repository to keep the noise correlation pipeline self-contained and reproducible.

Fenying Zang, Leiden University, 2026, f.zang@fsw.leidenuniv.nl

---
## Repository structure

- `scripts/`: analysis and plotting scripts
- `scripts/utils/`: reusable helper functions
- `data/`: local input data and derived tables
- `results/`: statistical outputs
- `figures/`: generated figures

## Main pipeline

Planned analysis steps:

1. Prepare metadata and inclusion tables
2. Load neuron pairs
3. Compute noise correlations
5. Run statistical analyses
6. Generate main and supplementary figures

## Installation & Setup

This project builds on the [IBL unified environment](https://github.com/int-brain-lab/iblenv).


## License

This project is licensed under the MIT License. See `LICENSE` for details.
