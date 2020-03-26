# Counterfactual metrics

This repository contains the implementation of the methods proposed in the paper [Counterfactual metrics](paper.pdf) by André Artelt and Barbara Hammer.

The proposed convex programs for computing counterfactual metrics of LVQ models are implemented in [metricchange.py](metricchange.py) (including a short usage example on a toy data set). In addition, the experiments for the drifting metric (as described in the paper) are implemented in [driftingmetric.py](driftingmetric.py).

The default solver is [SCS](https://github.com/cvxgrp/scs).

## Requirements

- Python3.6
- Packages as listed in `requirements.txt`

## License

MIT license - See [LICENSE.md](LICENSE)
