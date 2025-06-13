The Sinkhorn divergence algorithm I implemented is based on several key references in optimal transport theory:

  Primary References:

  1. Sinkhorn Divergence Definition:

  Genevay, A., Cuturi, M., Peyré, G., & Bach, F. (2018)"Stochastic Optimization for Large-scale Optimal Transport"Advances in Neural Information Processing Systems (NeurIPS) 2018arXiv:
  https://arxiv.org/abs/1707.08991

  This paper introduced the symmetric Sinkhorn divergence:
  S_λ(μ,ν) = OT_λ(μ,ν) - 0.5·OT_λ(μ,μ) - 0.5·OT_λ(ν,ν)

  2. Original Sinkhorn Algorithm:

  Sinkhorn, R., & Knopp, P. (1967)"Concerning nonnegative matrices and doubly stochastic matrices"Pacific Journal of Mathematics, 21(2), 343-348

  3. Entropic Regularized Optimal Transport:

  Cuturi, M. (2013)"Sinkhorn distances: Lightspeed computation of optimal transport"Advances in Neural Information Processing Systems (NIPS) 2013arXiv: https://arxiv.org/abs/1306.0895

  This seminal paper introduced the entropic regularization that makes the Sinkhorn algorithm practical:
  OT_λ(μ,ν) = min_π ⟨π, C⟩ + λ·H(π)

  Key Mathematical Concepts Implemented:

  1. Entropic Regularization:

  - Adds entropy term λ·H(π) to make the optimal transport problem smooth
  - Enables fast iterative solving via Sinkhorn scaling
  - Parameter λ controls regularization strength

  2. Sinkhorn Scaling Algorithm:

  - Alternating projections method for solving regularized optimal transport
  - Updates dual variables u and v iteratively:
  u[i] = log(μ[i]) - log(Σ_j K[i,j]·exp(v[j]))
  v[j] = log(ν[j]) - log(Σ_i K[i,j]·exp(u[i]))
  - Where K[i,j] = exp(-C[i,j]/λ) is the kernel matrix

  3. Cost Matrix:

  - Based on Euclidean distance between PMT positions
  - C[i,j] = ||pos_i - pos_j||_2 where positions are in 3D detector coordinates

  Modern Applications in Machine Learning:

  4. Applications to Point Clouds and Geometric Data:

  Feydy, J., Séjourné, T., Vialard, F. X., Amari, S. I., Trouvé, A., & Peyré, G. (2019)"Interpolating between optimal transport and MMD using Sinkhorn divergences"Artificial
  Intelligence and Statistics (AISTATS) 2019arXiv: https://arxiv.org/abs/1810.08278

  5. Computational Optimal Transport:

  Peyré, G., & Cuturi, M. (2019)"Computational optimal transport: With applications to data science"Foundations and Trends in Machine Learning, 11(5-6), 355-607Available:
  https://optimaltransport.github.io/

  Implementation Details:

  The algorithm implemented follows the log-domain stabilization approach to avoid numerical overflow, as described in:

  Schmitzer, B. (2019)"Stabilized sparse scaling algorithms for entropy regularized transport problems"SIAM Journal on Scientific Computing, 41(3), A1443-A1481

  Physical Motivation for PMT Applications:

  The use of optimal transport for comparing spatial distributions is well-established in:
  - Computer vision (comparing image histograms)
  - Neuroscience (comparing neural activity patterns)
  - Astronomy (comparing star field distributions)

  For optical flash matching in neutrino physics, this provides a natural way to compare the spatial patterns of predicted vs observed light, taking into account the 3D geometry of the
  PMT array rather than just comparing total photoelectron counts.

  The symmetric version ensures the divergence is:
  - Non-negative: S_λ(μ,ν) ≥ 0
  - Symmetric: S_λ(μ,ν) = S_λ(ν,μ)
  - Zero for identical distributions: S_λ(μ,μ) = 0
