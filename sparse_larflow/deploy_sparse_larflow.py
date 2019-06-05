from __future__ import print_function
import os,sys,time
import numpy as np
import torch
from larcv import larcv
from ublarcvapp import ublarcvapp
from ROOT import std
from sparsemodels import load_models
from load_cropped_sparse_dualflow import load_croppedset_sparse_dualflow_nomc
from crop_processor_cfg import fullsplit_processor_config

device = torch.device("cpu")
model  = load_models("dualflow_v1",weight_file="weights/dualflow/checkpoint.6850th.tar")
model.eval()
flow   = "dual"

#print(model)
validdata = ["testdata/larcvtruth-Run000002-SubRun002000.root"]

out = larcv.IOManager(larcv.IOManager.kWRITE, "stitched")
out.set_out_file( "test_sparseout_stitched.root" )
out.initialize()

dt_tot  = 0.0
dt_net  = 0.0
dt_data = 0.0
dt_result = 0.0

ttot = time.time()

# first create cfg file
processor_cfg = fullsplit_processor_config("wiremc","wiremc")
print(processor_cfg,file=open("cropflow_processor.cfg",'w'))
splitter = larcv.ProcessDriver( "ProcessDriver" )
splitter.configure( "cropflow_processor.cfg" )
io = splitter.io_mutable()
for inputfile in validdata:
    io.add_in_file(inputfile)
splitter.initialize()
nentries = io.get_n_entries()
nentries = 1
nimgs = 0

for ientry in xrange(nentries):
    io.read_entry(ientry)
    splitter.process_entry( ientry, False, False )
    ev_img = io.get_data(larcv.kProductImage2D, "wiremc")
    adc_v  = ev_img.Image2DArray()

    stitcher = ublarcvapp.UBSparseFlowStitcher( adc_v )
    ev_crops = io.get_data( larcv.kProductImage2D, "croppedadc" )
    crop_v   = ev_crops.Image2DArray()
    print( "[Entry {}] number of crops={}".format(ientry,crop_v.size()) )
    
    # get sparse numpy arrays
    tdata = time.time()
    data  = load_croppedset_sparse_dualflow_nomc(io)
    dt_data += time.time()-tdata

    ev_outdualflow_v = out.get_data( larcv.kProductSparseImage, "cropdualflow" )

    # torch tensors
    for iset,sparse_np in enumerate(data["pixadc"]):

        ncoords = sparse_np.shape[0]
        print("iset[{}] ncoords={}".format(iset,ncoords))
        
        # make tensor for coords (row,col,batch)
        coord_t  = torch.from_numpy( sparse_np[:,0:2].astype( np.int32 ) ).to(device)

        # tensor for src pixel adcs
        srcpix_t = torch.from_numpy( sparse_np[:,4].reshape( (ncoords,1) )  ).to(device)
        # tensor for target pixel adcs
        tarpix_flow1_t = torch.from_numpy( sparse_np[:,2].reshape( (ncoords,1) ) ).to(device)
        if flow=='dual':
            tarpix_flow2_t = torch.from_numpy( sparse_np[:,3].reshape( (ncoords,1) ) ).to(device)
        else:
            tarpix_flow2_t = None

        tnet = time.time()
        with torch.set_grad_enabled(False):
            predict1_t, predict2_t = model( coord_t, srcpix_t, tarpix_flow1_t, tarpix_flow2_t, 1 )
        dt_net += time.time()-tnet

        
        # back to numpy array
        tresult = time.time()
        
        meta_v = std.vector("larcv::ImageMeta")()
        yplane_meta = crop_v.at(iset*3+2).meta()
        meta_v.push_back( yplane_meta )
        meta_v.push_back( yplane_meta )        

        result_np = np.zeros( (ncoords,4), dtype=np.float32 )
        result_np[:,0:2] = sparse_np[:,0:2]
        result_np[:,2]   = predict1_t.features.numpy()[:,0]
        result_np[:,3]   = predict2_t.features.numpy()[:,0]

        # store raw result
        sparse_raw = larcv.sparseimg_from_ndarray( result_np, meta_v, larcv.msg.kDEBUG )
        ev_outdualflow_v.Append( sparse_raw )

        # prepare for stitcher
        result_np[:,2][ sparse_np[:,4]<10.0 ] = -1000
        result_np[:,3][ sparse_np[:,4]<10.0 ] = -1000
        sparse_result = larcv.sparseimg_from_ndarray( result_np, meta_v, larcv.msg.kDEBUG )
        stitcher.addSparseData( sparse_result, crop_v.at( iset*3+0 ).meta(), crop_v.at( iset*3+1 ).meta() )
        
        dt_result += time.time()-tresult
        nimgs += 1


    # store
    out_wire = out.get_data( larcv.kProductImage2D, "wire" )
    for p in xrange(3):
        out_wire.Append( adc_v.at(p) )
    out_y2u = out.get_data( larcv.kProductImage2D, "larflowy2u" )
    out_y2u.Append( stitcher._outimg_v.at(0) )
    out_y2v = out.get_data( larcv.kProductImage2D, "larflowy2v" )
    out_y2v.Append( stitcher._outimg_v.at(1) )

    out.set_id( ev_img.run(), ev_img.subrun(), ev_img.event() )

    out.save_entry()


dt_tot = time.time()-ttot

print( "Total run time: %.3f secs"%(dt_tot))
print( "  Data loading time: %.3f secs (%.3f secs/image)"%(dt_data, dt_data/nimgs))
print( "  Net running time: %.3f secs (%.3f secs/image)"%( dt_net,  dt_net/nimgs ))
print( "  Result conversion: %.3f secs (%.3f secs/image)"%( dt_result,  dt_result/nimgs ))

out.finalize()
splitter.finalize()

