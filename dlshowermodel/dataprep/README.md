# dataprep

Contains scripts to prepare training data.

##  `deploy_larmatchme.py`

This script takes in larcv and larlite information (e.g. from dlmerged files) and produces an HDF5 file that contains
the output of the larmatch network.

Example of how to run

```
python3 deploy_larmatchme.py -c config_larmatchme.yaml -w checkpoint.wise-capybara-107.101000.tar -ilcv ~/working/data/mcc9_v40a_dl_run3b_NC_pi0_overlay_CV/merged_dlreco_mcc9_v40a_dl_run3b_NC_pi0_overlay_CV_aa444faa-530a-4fd7-b43f-b501bc221880.root -ill ~/working/data/mcc9_v40a_dl_run3b_NC_pi0_overlay_CV/merged_dlreco_mcc9_v40a_dl_run3b_NC_pi0_overlay_CV_aa444faa-530a-4fd7-b43f-b501bc221880.root -o test.h5 --device cuda:0
```