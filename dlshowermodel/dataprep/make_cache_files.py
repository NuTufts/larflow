import os,sys
from dlshowermodel.data.larmatchhit_hdf5_reader import LArMatchHitHDF5Dataset


file_dirs = {"training":["output/outdir_mcc9_v13_bnb_nu_corsika_training/",
                         "output/outdir_mcc9_v13_bnbnue_corsika_training/"],
             "validation":["output/outdir_mcc9_v13_bnb_nu_corsika_validation/",
                           "output/outdir_mcc9_v13_bnbnue_corsika_validation/"]}

here_dir = "/cluster/tufts/wongjiradlabnu/twongj01/gen2/photon_analysis/ubdl/larflow/dlshowermodel/dataprep/"
for sample in file_dirs:
    sample_files = []
    for file_dir in file_dirs[sample]:
        flist = os.listdir( file_dir )
        for f in flist:
            sample_files.append( here_dir + "/" + file_dir + "/" + f )

    reader = LArMatchHitHDF5Dataset( file_paths=sample_files,
                                     file_has_training_labels=True )
    nfiles = len(sample_files)
    print(f'Make {sample} cache file: nfiles={nfiles}')
    reader.make_cache_file( f'dlshowermodel_{sample}_cache_file.txt' )
        


