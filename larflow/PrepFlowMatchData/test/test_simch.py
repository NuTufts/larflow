import os,sys

from larlite import larlite
from larcv import larcv
from larflow import larflow


#dlmerged_input = "/mnt/ddrive/data/ub_on_tufts/corsika_bnb_inue/dlmerged_coriska_bnb_nue_fileno000010.root"
#dlmerged_input = "/mnt/ddrive/data/ub_on_tufts/corsika_bnb_nu_pi0/dlmerged_coriska_bnb_nu_pi0_fileno000001.root"
#dlmerged_input = "/mnt/ddrive/data/ub_on_tufts/corsika_bnb_nu/dlmerged_coriska_bnb_nu_fileno000001.root"
dlmerged_input = "dlmerged.root"

start_entry = 0
end_entry = 0

simchmaker = larflow.prep.SimChTripletLabelMaker()
simchmaker._mcpixelmaker.set_dwire(1)
simchmaker._mcpixelmaker.set_drow(0)
#simchmaker._mcpixelmaker.set_verbosity(1)
simchmaker._mcpixelmaker.set_driftwc_source()
#simchmaker._mckpmaker.set_verbosity(1)
simchmaker.set_verbosity(1)
simchmaker.save_truth_tripletinfo( True )
simchmaker.open_hdf_file( "out_test_bnb_nu_pi0_dwire2_drow2_driftWC.h5" )

ioll = larlite.storage_manager( larlite.storage_manager.kREAD )
ioll.add_in_filename( dlmerged_input )
ioll.set_verbosity(2)
ioll.open()

iolcv = larcv.IOManager( larcv.IOManager.kREAD, "larcv" )
iolcv.add_in_file( dlmerged_input )
iolcv.set_verbosity(2)
iolcv.initialize()

for ientry in range(start_entry,end_entry+1):

  ioll.go_to(ientry)
  iolcv.read_entry(ientry)

  hdf_entry_prefix = f"/entry_{ientry}"
  simchmaker.process( ioll, iolcv )
  simchmaker._mckpmaker.printKeypoints()
  simchmaker.save_entry( hdf_entry_prefix )

simchmaker.close_hdf_file()
