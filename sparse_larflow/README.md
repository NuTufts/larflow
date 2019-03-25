LArFlow using Sparse Submanifold Convolutions

Notes on scripts


* `sparselarflow.py`: module with network
* `sparselarflowdata.py`: module with functions and classes for loading larflow data in sparse form
* `sparsify_data.py`: converts supera+larcvtruth information into sparsedata format for training
* `draw_sparsedata.py`: uses ROOT visualization tools to inspect output of `sparsify_data.py`
* `train_sparse_larflow.py`: training script for sparse larflow network
* `loss_sparse_larflow.py`: implements loss function. options for w/ and w/o 3d consistency loss term
* `func_intersect_ub.py`: functions for getting (y,z) intersection point from larflow predictions


Notes on folders

* `presparsedata`: scripts for applying `sparsify_data.py` onto a list of input files (on the Tufts cluster)
