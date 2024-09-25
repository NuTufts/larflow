import os,sys,random
from larmatch.data.larmatch_hdf5_reader import LArMatchHDF5Dataset

MAKE_TRAIN_CACHE = False
MAKE_VALID_CACHE = True

datasets = {"training":["/cluster/tufts/wongjiradlabnu/twongj01/gen2/photon_analysis/ubdl/larflow/larmatchnet/larmatch/prep/outdir_mcc9_v13_bnb_nu_corsika_training",
                        "/cluster/tufts/wongjiradlabnu/twongj01/gen2/photon_analysis/ubdl/larflow/larmatchnet/larmatch/prep/outdir_mcc9_v13_bnbnue_corsika_training"],
            "validation":["/cluster/tufts/wongjiradlabnu/twongj01/gen2/photon_analysis/ubdl/larflow/larmatchnet/larmatch/prep/outdir_mcc9_v13_bnb_nu_corsika_validation",
                          "/cluster/tufts/wongjiradlabnu/twongj01/gen2/photon_analysis/ubdl/larflow/larmatchnet/larmatch/prep/outdir_mcc9_v13_bnbnue_corsika_validation"]}

for dataset in datasets:
    if dataset=="training" and not MAKE_TRAIN_CACHE:
        continue
    if dataset=="validation" and not MAKE_VALID_CACHE:
        continue

    flist_out = []
    for outdir in datasets[dataset]:
        flist = os.listdir(outdir)
        for f in flist:
            if ".h5" in f:
                fpath = outdir+"/"+f
                flist_out.append(fpath)

    random.shuffle(flist_out)

    lm_dataset = LArMatchHDF5Dataset(file_paths=flist_out)
    lm_dataset.make_cache_file( "cache_list_larmatch_%s_dataset.txt"%(dataset) )

    

