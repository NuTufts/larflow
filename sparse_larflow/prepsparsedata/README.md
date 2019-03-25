# Prepare Sparsified LArFlow training files

Summary of workflow

* get list of input `larcvtruth` files. needs to contain ADC image and larflow truth information
* modify `inputlistname` check_jobs.py to set path to this list of `larcvtruth` files
* modify `workdir` to wherever you want the worker jobs to write files to
* use `run_check_jobs.sh` to run `check_jobs.py` in a container (because we need `PyROOT`)
* this will make the text file, `rerun_processing.list`
* now you can use `tufts_submit_sparsify.sh` to launch jobs

      sbatch tufts_submit_sparsify.sh

* after the jobs end, you can update the rerun list using `run_check_jobs.sh`
* repeat until all files processed
* then split successfully processed files into the `train` and `valid` set. For example, to take the first 1000 files for the train set

      cat processedok.list | sort | head -n 1000 > trainlist.txt

* then hadd using `submit_hadd_train.sh`