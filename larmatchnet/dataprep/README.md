# LArMatch/LArVoxel Training data

## Workflow

1. `run_kps_mcc9_v13_[bnb_nu/bnbnue]_corsika.sh`: Start by distilling truth from fully simulated events (from BNB+Coriska files) into `triplet` files.  This organizes truth data for both 2D pixels and 3D voxels along with the correspondence between each.
2. Then we further distill this truth information into the format used for training.

## Input files

We have two samples of BNB-coriska files. They care contained in the following file lists.

* `inputlists/mcc9_v13_bnb_nu_corsika.paired.list`: all neutrino interactions from the BNB flux (2863 files)
* `inputlists/mcc9_v13_bnbnue_corsika.paired.list`: only nue-CC interactions from the nue or nue-bar BNB flux (2461 files)

The lists contain a pair of files in each line.
The first file is the `larcv mctruth` file which contains the wire plane images along with images where each pixel encodes truth information.
The second file is the `larlite mcinfo` file which contains simulation truth metadata from Geant4.

## Triplet files

The output of the triplet code are provided in two lists:

* `inputlists/mcc9_v13_bnb_nu_coriska.triplet.list`: 563 files with 100 events each (except last file). about 56300 events total.


## Other notes


* Make list of input files in the `inputlists` folder.
* Adjust variables in `run_kps.sh`
* Submit jobs using `submit.sh`
* Make list of output files (using `find`)
* Split list into training set using `split_output_list.py`
* Use `hadd_submit.sh` to hadd
* Use `submit_cp.sh` to transfer to desginated training machine

