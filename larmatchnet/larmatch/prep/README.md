# data prep scripts

This folder contains the script

    make_larmatch_training_data_from_tripletfile.py

which is used to make the training data.

This folder also contains scripts to run jobs on the Tufts cluster.

## Description of the data

The input file to the above script are `larmatch triplet truth` files.
These files contain simulated MicroBooNE events where possible 3D points are formed.
The triplet truth files also contain necessary meta data (derived from the simulation) to make
different classes of labels.

For each 3D point, we associate to it truth labels pertaining to:
     * if the 3d point is real or false (aka a 'ghost'). this is the `larmatch` label.
     * what particle type made the 3d point. There are 7 classes. this is the `ssnet` label.
     * six different scores between 0 and 1, indicating how close the 3d point is to one of 6 different keypoints.
       this is the `keypoint` label.

In addition to the truth labels, the files also contain a larmatch, ssnet, and keypoint weight.
These weights are made based on the inverse frequency of the classes.
For the ssnet label, they also up-weight spacepoints near a neutrino vertex.

The files also contain information to make the "input data" to the larmatch network.
This data includes:
    * the wireplane images stored as sparse matrices made with pixel intensity threshold of 10
    * a list of "match triplets" which contain three indices for every 3D space point.
    There is an index for each wire plane. Each index points to an entry in the corresponding 2D wireplane sparse matrix.
    We use these indices in order to concatinate 2D feature vectors from the relevant location of each wire plane image.

## List of triplet files

In the `inputlists` folder, we have saved the location of `triplet truth` files that we have used to make the data.
There are two types of files: `bnbue` and `bnb_nu`.

The `bnbnue` files contain electron neutrino charged-current events.
The `bnb_nu` files contain all modes of neutrion-argon interactions,
with the flavor and energy of the incoming neutrino determine by the BNB flux at MicroBooNE.

## SLURM scripts

This folder also contains slurm

## Inspecting the contents

We provide (or will) python notebooks to inspect the contents of the output files.

## Using the data

In one folder above (`larmatch`) there is a file, `larmatch_dataset.py` which
implements a `torch.DataSet` class with which one can use to load the data for training.
At the end of the file, there is some code showing how to use it.
With a test file, one test/debug the class.

## Output Schema

The script above makes a ROOT file containing a single ROOT tree, `larmatchtrainingdata`.
The tree contains the following branches

```
    Example of loading and reading the data.

    Branches in the ROOT tree we are retrieveing

*............................................................................*
*Br    4 :coord_v   : Int_t coord_v_                                         *
*Entries :        5 : Total  Size=       2960 bytes  File Size  =        130 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.06     *
*............................................................................*
*Br    5 :coord_v.ndims : Int_t ndims[coord_v_]                              *
*Entries :        5 : Total  Size=        785 bytes  File Size  =        136 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.35     *
*............................................................................*
*Br    6 :coord_v.shape : vector<int> shape[coord_v_]                        *
*Entries :        5 : Total  Size=        935 bytes  File Size  =        223 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.50     *
*............................................................................*
*Br    7 :coord_v.data : vector<int> data[coord_v_]                          *
*Entries :        5 : Total  Size=    4476222 bytes  File Size  =    1261439 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   3.55     *
*............................................................................*
*Br    8 :feat_v    : Int_t feat_v_                                          *
*Entries :        5 : Total  Size=       2957 bytes  File Size  =        129 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.06     *
*............................................................................*
*Br    9 :feat_v.ndims : Int_t ndims[feat_v_]                                *
*Entries :        5 : Total  Size=        778 bytes  File Size  =        135 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.36     *
*............................................................................*
*Br   10 :feat_v.shape : vector<int> shape[feat_v_]                          *
*Entries :        5 : Total  Size=        868 bytes  File Size  =        201 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.36     *
*............................................................................*
*Br   11 :feat_v.data : vector<float> data[feat_v_]                          *
*Entries :        5 : Total  Size=    2238711 bytes  File Size  =     986523 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   2.27     *
*............................................................................*
*Br   12 :matchtriplet_v : Int_t matchtriplet_v_                             *
*Entries :        5 : Total  Size=       3093 bytes  File Size  =        137 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.06     *
*............................................................................*
*Br   13 :matchtriplet_v.ndims : Int_t ndims[matchtriplet_v_]                *
*Entries :        5 : Total  Size=        794 bytes  File Size  =        143 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.06     *
*............................................................................*
*Br   14 :matchtriplet_v.shape : vector<int> shape[matchtriplet_v_]          *
*Entries :        5 : Total  Size=        864 bytes  File Size  =        177 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.25     *
*............................................................................*
*Br   15 :matchtriplet_v.data : vector<int> data[matchtriplet_v_]            *
*Entries :        5 : Total  Size=   56939279 bytes  File Size  =   18744441 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   3.04     *
*............................................................................*
*Br   16 :larmatch_truth_v : Int_t larmatch_truth_v_                         *
*Entries :        5 : Total  Size=       3131 bytes  File Size  =        136 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.08     *
*............................................................................*
*Br   17 :larmatch_truth_v.ndims : Int_t ndims[larmatch_truth_v_]            *
*Entries :        5 : Total  Size=        788 bytes  File Size  =        132 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.01     *
*............................................................................*
*Br   18 :larmatch_truth_v.shape : vector<int> shape[larmatch_truth_v_]      *
*Entries :        5 : Total  Size=        818 bytes  File Size  =        150 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.09     *
*............................................................................*
*Br   19 :larmatch_truth_v.data : vector<int> data[larmatch_truth_v_]        *
*Entries :        5 : Total  Size=        813 bytes  File Size  =        149 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.09     *
*............................................................................*
*Br   20 :larmatch_weight_v : Int_t larmatch_weight_v_                       *
*Entries :        5 : Total  Size=       3166 bytes  File Size  =        137 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.08     *
*............................................................................*
*Br   21 :larmatch_weight_v.ndims : Int_t ndims[larmatch_weight_v_]          *
*Entries :        5 : Total  Size=        795 bytes  File Size  =        133 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.01     *
*............................................................................*
*Br   22 :larmatch_weight_v.shape : vector<int> shape[larmatch_weight_v_]    *
*Entries :        5 : Total  Size=        825 bytes  File Size  =        151 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.09     *
*............................................................................*
*Br   23 :larmatch_weight_v.data : vector<float> data[larmatch_weight_v_]    *
*Entries :        5 : Total  Size=        820 bytes  File Size  =        150 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.09     *
*............................................................................*
*Br   24 :ssnet_truth_v : Int_t ssnet_truth_v_                               *
*Entries :        5 : Total  Size=       3074 bytes  File Size  =        133 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.08     *
*............................................................................*
*Br   25 :ssnet_truth_v.ndims : Int_t ndims[ssnet_truth_v_]                  *
*Entries :        5 : Total  Size=        767 bytes  File Size  =        129 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.01     *
*............................................................................*
*Br   26 :ssnet_truth_v.shape : vector<int> shape[ssnet_truth_v_]            *
*Entries :        5 : Total  Size=        797 bytes  File Size  =        147 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.09     *
*............................................................................*
*Br   27 :ssnet_truth_v.data : vector<int> data[ssnet_truth_v_]              *
*Entries :        5 : Total  Size=        792 bytes  File Size  =        146 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.09     *
*............................................................................*
*Br   28 :ssnet_weight_v : Int_t ssnet_weight_v_                             *
*Entries :        5 : Total  Size=       3109 bytes  File Size  =        134 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.08     *
*............................................................................*
*Br   29 :ssnet_weight_v.ndims : Int_t ndims[ssnet_weight_v_]                *
*Entries :        5 : Total  Size=        774 bytes  File Size  =        130 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.01     *
*............................................................................*
*Br   30 :ssnet_weight_v.shape : vector<int> shape[ssnet_weight_v_]          *
*Entries :        5 : Total  Size=        804 bytes  File Size  =        148 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.09     *
*............................................................................*
*Br   31 :ssnet_weight_v.data : vector<float> data[ssnet_weight_v_]          *
*Entries :        5 : Total  Size=        799 bytes  File Size  =        147 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.09     *
*............................................................................*
*Br   32 :kp_truth_v : Int_t kp_truth_v_                                     *
*Entries :        5 : Total  Size=       3033 bytes  File Size  =        130 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.08     *
*............................................................................*
*Br   33 :kp_truth_v.ndims : Int_t ndims[kp_truth_v_]                        *
*Entries :        5 : Total  Size=        746 bytes  File Size  =        126 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.01     *
*............................................................................*
*Br   34 :kp_truth_v.shape : vector<int> shape[kp_truth_v_]                  *
*Entries :        5 : Total  Size=        776 bytes  File Size  =        144 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.09     *
*............................................................................*
*Br   35 :kp_truth_v.data : vector<float> data[kp_truth_v_]                  *
*Entries :        5 : Total  Size=        771 bytes  File Size  =        143 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.09     *
*............................................................................*
*Br   36 :kp_weight_v : Int_t kp_weight_v_                                   *
*Entries :        5 : Total  Size=       3052 bytes  File Size  =        131 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.08     *
*............................................................................*
*Br   37 :kp_weight_v.ndims : Int_t ndims[kp_weight_v_]                      *
*Entries :        5 : Total  Size=        753 bytes  File Size  =        127 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.01     *
*............................................................................*
*Br   38 :kp_weight_v.shape : vector<int> shape[kp_weight_v_]                *
*Entries :        5 : Total  Size=        783 bytes  File Size  =        145 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.09     *
*............................................................................*
*Br   39 :kp_weight_v.data : vector<float> data[kp_weight_v_]                *
*Entries :        5 : Total  Size=        778 bytes  File Size  =        144 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.09     *
*............................................................................*
```