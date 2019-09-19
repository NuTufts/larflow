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


## `deploy_sparse_larflow`

Run (reco)

    ./deploy_sparse_larflow.py -r -olcv test_larcv.root -oll test_larlite.root  -w weights/dualflow/dualflow.classvec.checkpoint.242000th.tar -adc wiremc -ch wiremc testdata/larcvtruth-Run000002-SubRun002000.ro

Run without hits, save cropped output and save trueflow

    ./deploy_sparse_larflow.py -r -olcv test_larcv.root -oll test_larlite.root  -w weights/dualflow/dualflow.classvec.checkpoint.242000th.tar -adc wiremc -ch wiremc --no-flowhits --save-trueflow-crops testdata/larcvtruth-Run000002-SubRun002000.root