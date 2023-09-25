# Partial Optimal Transport Registration

This repository provides example code for non-linear point-cloud registration using partial optimal transport. The code demonstrates both 3D and 2D numerical examples as described in the accompanying paper. The implementation is done using Python, numpy, torch, and scipy, without the need for any specialized dependencies.

## Installation
To set up your environment, you'll need to install [torch](https://pytorch.org), [numba](https://numba.pydata.org), and [PythonOT](https://pythonot.github.io).

## Repository Structure
- `lib/lib_ot.py`: Includes variants of Optimal Transport (OT) and Partial Optimal Transport (OPT) solvers. Algorithms such as Network Simplex[1,2,3], Sinkhorn[4,5], Sliced OT[6], and Sliced OPT[7] are provided. 
- `lib/lib_shape.py`: new methods proposed in this paper, namely OPT-RBF, OPT-TPS, SOPT-RBF, and SOPT-TPS, as well as the four balanced Wasserstein Procrustes methods (OT-RBF, OT-TPS, SOT-RBF, SOT-TPS).
In addition, we provide the sliced-OT/OPT gradient flow[8] and OT/OPT barycentric projection[9, 10,11].

- `lib/cpd.py`: Contains implementation of the Coherent Point Drift (CPD) method[12].

## Running Experiments
- Execute `3D-experiment.ipynb` to reproduce the numerical results for the 3D experiment.
- Execute `2D-experiment.ipynb` to reproduce the numerical results for the 2D fish experiment.

- references:
[1] Bonneel, N., Van De Panne, M., Paris, S., & Heidrich, W. (2011, December). Displacement interpolation using Lagrangian mass transport. In ACM Transactions on Graphics (TOG) (Vol. 30, No. 6, p. 158). ACM.
[2] Caffarelli, L. A., & McCann, R. J. (2010) Free boundaries in optimal transport and Monge-Ampere obstacle problems. Annals of mathematics, 673-730.
[3] Chapel, L., Alaya, M., Gasso, G. (2020). “Partial Optimal Transport with Applications on Positive-Unlabeled Learning”. NeurIPS.

[4]
Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016). Scaling algorithms for unbalanced transport problems. arXiv preprint arXiv:1607.05816.

[5]
Frogner C., Zhang C., Mobahi H., Araya-Polo M., Poggio T. : Learning with a Wasserstein Loss, Advances in Neural Information Processing Systems (NIPS) 2015

[6] Bonneel, Nicolas, et al. “Sliced and radon wasserstein barycenters of measures.” Journal of Mathematical Imaging and Vision 51.1 (2015): 22-45

[7] Bai, Y., Schmitzer, B., Thorpe, M., & Kolouri, S. (2023). Sliced optimal partial transport. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 13681-13690).

[8] Bonet, C., Courty, N., Septier, F., & Drumetz, L. (2021). Sliced-Wasserstein gradient flows. arXiv preprint arXiv:2110.10972.

[9] Wang, W., Slepčev, D., Basu, S., Ozolek, J. A., & Rohde, G. K. (2013). A linear optimal transportation framework for quantifying and visualizing variations in sets of images. International journal of computer vision, 101, 254-269.

[10] Bai, Y., Medri, I. V., Martin, R. D., Shahroz, R., & Kolouri, S. (2023, July). Linear optimal partial transport embedding. In International Conference on Machine Learning (pp. 1492-1520). PMLR.

[11] Ambrosio, L., Gigli, N., & Savaré, G. (2005). Gradient Flows: In Metric Spaces and in the Space of Probability Measures.

[12] A. Myronenko and X. Song, "Point Set Registration: Coherent Point Drift," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 32, no. 12, pp. 2262-2275, Dec. 2010, doi: 10.1109/TPAMI.2010.46.

