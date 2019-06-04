import os,sys,time
import numpy as np
import torch
from larcv import larcv
from sparsemodels import load_models
from load_wholeview_sparse_dualflow import load_wholeview_sparse_dualflow

device = torch.device("cpu")
model  = load_models("dualflow_v1",weight_file="weights/dualflow/checkpoint.6850th.tar")
model.eval()
flow   = "dual"

print model
validdata = []
# iovalid = load_larflow_larcvdata( "valid", validdata,
#                                   BATCHSIZE_VALID, NWORKERS_VALID,
#                                   producer_name="sparsecropdual",
#                                   nflows=len(flowdirs),
#                                   tickbackward=TICKBACKWARD,
#                                   readonly_products=None )

# whole view
io = larcv.IOManager(larcv.IOManager.kREAD,"wholeview",larcv.IOManager.kTickBackward)
io.add_in_file( "testdata/larcvtruth-Run000002-SubRun002000.root" )
io.initialize()

dt_tot  = 0.0
dt_net  = 0.0
dt_data = 0.0

ttot = time.time()

nentries = io.get_n_entries()
nentries = 10
nimgs = 0
for ientry in xrange(nentries):
    io.read_entry(ientry)

    # get sparse numpy arrays
    tdata = time.time()
    data  = load_wholeview_sparse_dualflow(io)
    dt_data += time.time()-tdata

    # torch tensors
    ncoords = data["pixadc"].shape[0]

    # make tensor for coords (row,col,batch)
    coord_t = torch.zeros( (ncoords,3), dtype=torch.int ).to(device)

    # tensor for src pixel adcs
    srcpix_t = torch.zeros( (ncoords,1), dtype=torch.float).to(device)
    # tensor for target pixel adcs
    tarpix_flow1_t = torch.zeros( (ncoords,1), dtype=torch.float).to(device)
    if flow=='dual':
        tarpix_flow2_t = torch.zeros( (ncoords,1), dtype=torch.float).to(device)
    else:
        tarpix_flow2_t = None

    tnet = time.time()
    with torch.set_grad_enabled(False):
        predict1_t, predict2_t = model( coord_t, srcpix_t, tarpix_flow1_t, tarpix_flow2_t, 1 )
    print predict1_t.features.size()
    dt_net += time.time()-tnet

    nimgs += 1

dt_tot = time.time()-ttot

print "Total run time: %.3f secs"%(dt_tot)
print "  Data loading time: %.3f secs (%.3f secs/image)"%(dt_data, dt_data/nimgs)
print "  Net running time: %.3f secs (%.3f secs/image)"%( dt_net,  dt_net/nimgs )

