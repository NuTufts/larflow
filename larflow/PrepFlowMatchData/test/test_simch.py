import os,sys

from larlite import larlite
from larcv import larcv
from larflow import larflow


#dlmerged_input = "/mnt/ddrive/data/ub_on_tufts/corsika_bnb_inue/dlmerged_coriska_bnb_nue_fileno000010.root"
dlmerged_input = "/mnt/ddrive/data/ub_on_tufts/corsika_bnb_nu_pi0/dlmerged_coriska_bnb_nu_pi0_fileno000001.root"

ENTRY=0

simchmaker = larflow.prep.SimChTripletLabelMaker()
simchmaker.set_verbosity(1)

ioll = larlite.storage_manager( larlite.storage_manager.kREAD )
ioll.add_in_filename( dlmerged_input )
ioll.set_verbosity(2)
ioll.open()

iolcv = larcv.IOManager( larcv.IOManager.kREAD, "larcv" )
iolcv.add_in_file( dlmerged_input )
iolcv.set_verbosity(2)
iolcv.initialize()

ioll.go_to(ENTRY)
iolcv.read_entry(ENTRY)

print(simchmaker)

simchmaker.process( ioll, iolcv )

simchmaker.export_as_hdf("out_test.h5")
