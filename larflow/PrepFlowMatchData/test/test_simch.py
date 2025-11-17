import os,sys

from larlite import larlite
from larcv import larcv
from larflow import larflow


dlmerged_input = "/mnt/ddrive/data/ub_on_tufts/corsika_bnb_inue/dlmerged_coriska_bnb_nue_fileno000010.root"

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

ioll.go_to(0)
iolcv.read_entry(0)

print(simchmaker)

simchmaker.process( ioll, iolcv )
