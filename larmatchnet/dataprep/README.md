## Workflow

* Make list of input files in the `inputlists` folder.
* Adjust variables in `run_kps.sh`
* Submit jobs using `submit.sh`
* Make list of output files (using `find`)
* Split list into training set using `split_output_list.py`
* Use `hadd_submit.sh` to hadd
* Use `submit_cp.sh` to transfer to desginated training machine

