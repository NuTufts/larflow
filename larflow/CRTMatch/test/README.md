# CRTMatch test/deploy scripts

Scripts for deploying and testing CRT Matching Algorithms.

## `run_crtmatchmanager.py`

usage example:

```
python run_crtmatchmanager.py --input-dlmerged mcc9_v29e_run3_G1_extbnb_merged_dlana_ffddd628-f39a-4db4-a797-7dc323ddc0c6.root --input-larmatch larmatch_paf_ffddd628-f39a-4db4-a797-7dc323ddc0c6_larlite.root -o test.root
```

```
usage: Run CRT-MATCHING algorithms [-h] -dl INPUT_DLMERGED -lm INPUT_LARMATCH
                                   -o OUTPUT [-n NUM_ENTRIES] [-e START_ENTRY]

optional arguments:
  -h, --help            show this help message and exit
  -dl INPUT_DLMERGED, --input-dlmerged INPUT_DLMERGED
                        DL merged file
  -lm INPUT_LARMATCH, --input-larmatch INPUT_LARMATCH
                        larmatch output
  -o OUTPUT, --output OUTPUT
                        Output file stem
  -n NUM_ENTRIES, --num-entries NUM_ENTRIES
                        Number of entries to run
  -e START_ENTRY, --start-entry START_ENTRY
                        Starting entry
```


You can visualize the output using `vis_cmm.py`

Example:

![Image](example_vis_cmm_run3g1.png?raw=true)




## `run_crttrack_match.py

This is *deprecated*, using thru-mu only reconstruction code
that tries to avoid using larmatch-based reco to find tracks.
Not accurate or efficiency enough.

Paired script: `vis_crttrack_match.py`.