=====
About
=====

What is cilantro?
=================
**cilantro** is a lean and fast C++ library for working with point cloud data, with emphasis given to the 3D case. It includes efficient implementations for a variety of common operations, providing a clean API and attempting to minimize the amount of boilerplate code. The library is extensively templated, enabling operations on point data of arbitrary numerical type and dimensionality (where applicable) and featuring a modular/extensible design of the more complex procedures, while, at the same time, providing convenience aliases/wrappers for the most common cases. A high-level description of **cilantro** can be found in our `technical report`_.

Features
========

* **Core operations:**

    - General dimension kd-trees (using bundled nanoflann_)
    - Surface normal and curvature estimation from raw point clouds
    - General dimension grid-based point cloud resampling
    - Principal Component Analysis
    - Basic I/O utilities for 3D point clouds (in PLY format, using bundled tinyply_) and Eigen matrices
    - RGBD images to/from point cloud utility functions

* **Convex hulls and spatial reasoning tools:**

    - A general dimension convex polytope representation that is computed (using bundled Qhull_) from either vertex or half-space intersection input and allows for easy switching between the respective representations
    - A representation of generic (general dimension) space regions as unions of convex polytopes that implements set operations

* **Clustering:**

    - General dimension k-means clustering that supports all distance metrics supported by nanoflann
    - Spectral clustering based on various graph Laplacian types (using bundled Spectra)
    - Mean-shift clustering with custom kernel support
    - Connected component based point cloud segmentation that supports arbitrary point-wise similarity functions

* **Geometric registration:**

    - Multiple generic Iterative Closest Point implementations that support arbitrary correspondence search methods in arbitrary point feature spaces for:

        * **Rigid** alignment under the point-to-point metric (general dimension), point-to-plane metric (3D), or any combination thereof
        * **Non-rigid** alignment of 3D point sets, by means of a robustly regularized, locally rigid warp field, under any combination of the point-to-point and point-to-plane metrics; implementations for both *densely* and *sparsely* (similarly to DynamicFusion_) supported warp fields are provided

* **Robust model estimation:**

    - A RANSAC estimator template and instantiations thereof for general dimension:

        * Robust hyperplane estimation
        * Rigid point cloud registration given noisy correspondences

* **Visualization:**

    - Classical Multidimensional Scaling (using bundled Spectra_ for eigendecompositions)
    - A powerful, extensible, and easy to use 3D visualizer

.. _nanoflann: https://github.com/jlblancoc/nanoflann
.. _Spectra: https://github.com/yixuan/spectra
.. _tinyply: https://github.com/ddiakopoulos/tinyply
.. _Qhull: http://www.qhull.org/
.. _technical report: https://arxiv.org/abs/1807.00399
.. _DynamicFusion: http://grail.cs.washington.edu/projects/dynamicfusion/
