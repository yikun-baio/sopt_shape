# Partial Transport for Point-Cloud Registration

This repository contains code and experiments for the paper [Partial Transport for Point-Cloud Registration](https://arxiv.org/abs/2309.15787). The project studies non-rigid point-cloud registration with optimal partial transport (OPT) and sliced optimal partial transport (SOPT), especially when source and target point clouds contain noise, outliers, or missing/extra points.

![Point-cloud registration examples](assets/registration_overview.png)

The main contribution implemented here is a family of OPT/SOPT-based non-rigid registration methods using RBF and TPS deformation models. The repository also includes balanced OT/SOT variants and classical baselines for comparison.

## Installation

Create the conda environment from the included YAML file:

```bash
conda env create -f environment.yml
conda activate sopt-shape
```

The environment is intentionally small. The core packages are PyTorch, NumPy, Numba, and POT/PythonOT (`import ot`). The YAML also includes SciPy because several OT/registration utilities call it directly, Matplotlib because the library files import plotting utilities, and `notebook/ipykernel` for running the experiment notebooks.

## Repository Structure

### Our Method

- `lib/lib_shape.py`: main point-cloud registration implementation. This file contains the proposed OPT/SOPT registration methods:
  - `OPT-RBF`
  - `OPT-TPS`
  - `SOPT-RBF`
  - `SOPT-TPS`
- `lib/lib_shape.py` also includes the balanced Wasserstein Procrustes variants used for comparison:
  - `OT-RBF`
  - `OT-TPS`
  - `SOT-RBF`
  - `SOT-TPS`
- The same file includes supporting registration utilities such as RBF/TPS kernels, rigid initialization, barycentric projection, gradient-flow routines, visualization, and error computation.

### Algorithms and Utilities

- `lib/lib_ot.py`: OT and OPT solvers, including EMD/network-simplex style wrappers, Sinkhorn variants, partial OT, 1D OPT, and sliced OPT helpers.
- `lib/sliced_opt.py`: sliced OT/OPT projection utilities and correspondence recovery.
- `lib/library.py`: shared numerical utilities, cost-matrix functions, index recovery helpers, Gaussian-mixture sampling, and partial OT helpers.
- `opt1d.cpp`: C++ implementation/support code for one-dimensional optimal partial transport.
- `lib/tools.py`: small project helper functions.
- `lib/fish.py`: fish-shape data transformations, noise generation, and subsampling helpers.

### Baselines

- `lib/cpd.py`: Coherent Point Drift (CPD) baseline.
- `lib/icp_ffd.py`: ICP with Free-Form Deformation (ICP-FFD) baseline.

### Experiments and Data

- `2D_experiment.ipynb`: reproduces the 2D fish experiments.
- `3D-experiment.ipynb`: reproduces the 3D point-cloud experiments.
- `OT_fish_example.ipynb`: smaller fish-shape OT/OPT example.
- `data_generate.ipynb`: data construction and noise generation.
- `data/`: prepared point-cloud datasets used by the notebooks.
- `results/`: generated or saved experiment outputs and figures.
- `assets/registration_overview.png`: README figure copied from the arXiv source package for the paper.

## Running Experiments

Start Jupyter from the activated environment:

```bash
jupyter notebook
```

Then run:

- `2D_experiment.ipynb` for the 2D fish registration experiment.
- `3D-experiment.ipynb` for the 3D registration experiment.
- `OT_fish_example.ipynb` for a compact OT/OPT fish-shape example.

Some notebook cells save intermediate models and figures into `results/`. If you run on CPU, you may need to change notebook variables such as `device='cuda:1'` to `device='cpu'`.

## Paper

- arXiv page: <https://arxiv.org/abs/2309.15787>
- PDF: <https://arxiv.org/pdf/2309.15787>

If this code is useful for your work, please cite the paper:

```bibtex
@article{bai2023partial,
  title={Partial Transport for Point-Cloud Registration},
  author={Bai, Yikun and Tran, Huy and Damelin, Steven B. and Kolouri, Soheil},
  journal={arXiv preprint arXiv:2309.15787},
  year={2023}
}
```

## References

[1] Bonneel, N., Van De Panne, M., Paris, S., & Heidrich, W. (2011). Displacement interpolation using Lagrangian mass transport. ACM Transactions on Graphics.

[2] Caffarelli, L. A., & McCann, R. J. (2010). Free boundaries in optimal transport and Monge-Ampere obstacle problems. Annals of Mathematics.

[3] Chapel, L., Alaya, M., & Gasso, G. (2020). Partial Optimal Transport with Applications on Positive-Unlabeled Learning. NeurIPS.

[4] Chizat, L., Peyre, G., Schmitzer, B., & Vialard, F. X. (2016). Scaling algorithms for unbalanced transport problems. arXiv:1607.05816.

[5] Frogner, C., Zhang, C., Mobahi, H., Araya-Polo, M., & Poggio, T. (2015). Learning with a Wasserstein Loss. NeurIPS.

[6] Bonneel, N., et al. (2015). Sliced and Radon Wasserstein Barycenters of Measures. Journal of Mathematical Imaging and Vision.

[7] Bai, Y., Schmitzer, B., Thorpe, M., & Kolouri, S. (2023). Sliced Optimal Partial Transport. CVPR.

[8] Bonet, C., Courty, N., Septier, F., & Drumetz, L. (2021). Sliced-Wasserstein Gradient Flows. arXiv:2110.10972.

[9] Wang, W., Slepcev, D., Basu, S., Ozolek, J. A., & Rohde, G. K. (2013). A linear optimal transportation framework for quantifying and visualizing variations in sets of images. IJCV.

[10] Bai, Y., Medri, I. V., Martin, R. D., Shahroz, R., & Kolouri, S. (2023). Linear Optimal Partial Transport Embedding. ICML.

[11] Ambrosio, L., Gigli, N., & Savare, G. (2005). Gradient Flows: In Metric Spaces and in the Space of Probability Measures.

[12] Myronenko, A., & Song, X. (2010). Point Set Registration: Coherent Point Drift. IEEE TPAMI.
