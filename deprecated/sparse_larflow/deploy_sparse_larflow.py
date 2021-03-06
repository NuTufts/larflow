#!/usr/bin/env python
from __future__ import print_function
import os,sys,time
import argparse

parser = argparse.ArgumentParser(description='Process ADC images through LArFlow and LArFlow Hit maker.')
parser.add_argument('inputfiles',type=str,nargs='+',help='list of input files: either supera (reco only) or larcvtruth (reco+truth)')
parser.add_argument('-olcv','--outfile-larcv',  required=True,type=str,help='name of output LArCV file (holds crops of sparseimage)')
parser.add_argument('-oll', '--outfile-larlite',required=True,type=str,help='name of output larlite file (holds larflow3dhits)')
parser.add_argument('-d','--weight-dir',required=False,default='.',help='directory containing network weight files')
parser.add_argument('-w','--weight-file',required=True,type=str,help='network checkpoint or weight file')
parser.add_argument('-r','--overwrite',action="store_true",help='allow overwriting of output files')
parser.add_argument('-c','--config',default='ubcroptrueflow.cfg',type=str,help='configuration file for splitter and cropper')
parser.add_argument('-mc','--has-mc',action="store_true",default=False,help='file has truth information to process')
parser.add_argument("-mo","--model",default="dualflow_classvec_v2",type=str,help='Model Name to load') 
parser.add_argument('-adc','--adc-producer',default='wire',type=str,help='producer name for full ADC images')
parser.add_argument('-ch','--chstatus-producer',default='wire',type=str,help='producer name for (larcv) ChStatus')
parser.add_argument('--no-flowhits',action="store_true",default=False,help="Make flow hits")
parser.add_argument('--save-trueflow-crops',action="store_true",default=False,help="Store True Flow Crops")
parser.add_argument("-n", "--num",default=-1,type=int,help="Number of entries to run. <0 means run all")

def deploy_sparselarflow_on_files( larcv_outfile, larlite_outfile, filelist, weightfile,
                                   model_name="dualflow_classvec_v2",
                                   adc_producer="wire",
                                   chstatus_producer='wire',
                                   cropper_cfg="cropflow_processor.cfg",
                                   flow="dual", devicename="cpu",
                                   run_reco_flowhits=True,
                                   run_truth_flowhits=True,                                   
                                   save_full_adc=False,
                                   save_cropped_adc=False,
                                   save_cropped_trueflow=False,                                  
                                   run_stitcher=False,
                                   has_mc=False,
                                   threshold=10.0,
                                   maxentries=-1 ):

    import numpy as np
    import torch
    from larlite import larlite
    from larcv import larcv
    from ublarcvapp import ublarcvapp
    from larflow import larflow
    from ROOT import std

    from sparsemodels import load_models
    from load_cropped_sparse_dualflow import load_croppedset_sparse_dualflow_nomc
    
    device = torch.device(devicename)
    model  = load_models(model_name,weight_file=weightfile, device=devicename )
    model.eval()
    
    out = larcv.IOManager(larcv.IOManager.kWRITE, "stitched")
    out.set_out_file( larcv_outfile )
    out.initialize()

    out_ll = larlite.storage_manager(larlite.storage_manager.kWRITE)
    out_ll.set_out_filename(larlite_outfile)
    out_ll.open()

    dt_tot  = 0.0
    dt_net  = 0.0     # running the network
    dt_data = 0.0     # preparing data (split/crop)
    dt_aten = 0.0     # turn data into torch tensors
    dt_flow = 0.0     # making flow
    dt_result = 0.0   # preparing output images

    ttot = time.time()
    
    # first create cfg file if does not exist
    if not os.path.exists( cropper_cfg ):
        print("Writing new copper config: ",cropper_cfg)
        from crop_processor_cfg import fullsplit_processor_config
        f = open(cropper_cfg,'w')
        f.write( fullsplit_processor_config(adc_producer,chstatus_producer) )
        f.close()
        
    splitter = larcv.ProcessDriver( "ProcessDriver" )
    print("CONFIGURE SPLITTER: ",cropper_cfg)
    splitter.configure( cropper_cfg )

    # add files to iomanager
    io = splitter.io_mutable()

    if type(filelist) is str:
        filelist = [filelist]
        
    for inputfile in filelist:
        io.add_in_file(inputfile)
        
    # initialize splitter
    splitter.initialize()
    nentries = io.get_n_entries()
    if maxentries>0 and maxentries<nentries:
        nentries = maxentries

    nimgs = 0
    nevents = 0
    for ientry in xrange(nentries):

        tdata = time.time()
        io.read_entry(ientry)
        ev_img = io.get_data(larcv.kProductImage2D, adc_producer)

        run    = ev_img.run()
        subrun = ev_img.subrun()
        event  = ev_img.event()

        print( "[Entry {}] {}".format(ientry,(run,subrun,event)) )
        
        adc_v  = ev_img.Image2DArray()
        adc_copy_v = std.vector("larcv::Image2D")()
        for i in xrange(adc_v.size()):
            adc_copy_v.push_back( adc_v.at(i) )
            
        splitter.process_entry( ientry, False, False )        

        if run_stitcher:
            stitcher = ublarcvapp.UBSparseFlowStitcher( adc_v )
            
        ev_crops = io.get_data( larcv.kProductImage2D, "croppedadc" )
        crop_v   = ev_crops.Image2DArray()
        print("  number of crops: {}".format(crop_v.size()))
    
        # get sparse numpy arrays
        data  = load_croppedset_sparse_dualflow_nomc(io)
        dt_data += time.time()-tdata

        # container for network output
        ev_outdualflow_v = out.get_data( larcv.kProductSparseImage, "cropdualflow" )

        # torch tensors
        for iset,sparse_np in enumerate(data["pixadc"]):

            taten = time.time()
            
            ncoords = sparse_np.shape[0]
            print("deploy net: iset[{}] ncoords={}".format(iset,ncoords))
        
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

            dt_aten += time.time()-taten

            # Run NETWORK
            tnet = time.time()
            with torch.set_grad_enabled(False):
                predict1_t, predict2_t = model( coord_t, srcpix_t, tarpix_flow1_t, tarpix_flow2_t, 1 )
            dt_net += time.time()-tnet
            #print("predict1_t shape",predict1_t.features.shape)
            
            # convert class vector output back to flow
            # find max, then subtract off source pix
            if model_name in ['dualflow_classvec_v2']:
                # get arg max
                maxcol1 = torch.argmax( predict1_t.features.detach(), 1 )
                maxcol2 = torch.argmax( predict2_t.features.detach(), 1 )
                # subtract source column
                flowout1_t = (maxcol1.type(torch.FloatTensor) - coord_t[:,1].type(torch.FloatTensor)).reshape( (ncoords,1) )
                flowout2_t = (maxcol2.type(torch.FloatTensor) - coord_t[:,1].type(torch.FloatTensor)).reshape( (ncoords,1) )
            else:
                flowout1_t = predict1_t.features
                flowout2_t = predict2_t.features
                
            
            # back to numpy array
            tresult = time.time()
        
            meta_v = std.vector("larcv::ImageMeta")()
            yplane_meta = crop_v.at(iset*3+2).meta()
            meta_v.push_back( yplane_meta )
            meta_v.push_back( yplane_meta )        

            result_np = np.zeros( (ncoords,4), dtype=np.float32 )
            result_np[:,0:2] = sparse_np[:,0:2]
            result_np[:,2]   = flowout1_t.detach().cpu().numpy()[:,0]
            result_np[:,3]   = flowout2_t.detach().cpu().numpy()[:,0]

            # store raw result
            sparse_raw = larcv.sparseimg_from_ndarray( result_np, meta_v, larcv.msg.kDEBUG )
            ev_outdualflow_v.Append( sparse_raw )

            # prepare for stitcher
            if run_stitcher:
                result_np[:,2][ sparse_np[:,4]<10.0 ] = -1000
                result_np[:,3][ sparse_np[:,4]<10.0 ] = -1000
                sparse_result = larcv.sparseimg_from_ndarray( result_np, meta_v, larcv.msg.kDEBUG )
                stitcher.addSparseData( sparse_result, crop_v.at( iset*3+0 ).meta(), crop_v.at( iset*3+1 ).meta() )
        
            dt_result += time.time()-tresult
            nimgs += 1

        # make flow hits
        # --------------
        tflow = time.time()
        if run_reco_flowhits:
            print("Make Reco Flow Hits")
            larflowhits_v = larflow.makeFlowHitsFromSparseCrops(adc_v, ev_outdualflow_v.SparseImageArray(),
                                                                threshold, "ubcroptrueflow.cfg", larcv.msg.kINFO )

        if has_mc and run_truth_flowhits:
            print("Make Truth Flow Hits")
            ev_chstatus = io.get_data( larcv.kProductChStatus, chstatus_producer )
            ev_trueflow = io.get_data( larcv.kProductImage2D, "larflow" )
            trueflowhits_v = larflow.makeTrueFlowHitsFromWholeImage( adc_v, ev_chstatus, ev_trueflow.Image2DArray(), threshold,
                                                                     "ubcroptrueflow.cfg", larcv.msg.kINFO )
                                                                     
        
        dt_flow += time.time()-tflow
        
        # store
        # --------
        # full image
        tresult = time.time()
        
        if save_full_adc:
            out_wire = out.get_data( larcv.kProductImage2D, "wire" )
            for p in xrange(3):
                out_wire.Append( adc_v.at(p) )
        
        # cropped image
        if save_cropped_adc:
            out_crop = out.get_data( larcv.kProductImage2D, "cropadc" )
            for iimg in xrange(crop_v.size()):
                out_crop.Append( crop_v.at(iimg) )
            print("saved ",crop_v.size()," adc crops")

        if save_cropped_trueflow:
            ev_trueflow_crops  = io.get_data(  larcv.kProductImage2D, "croppedflow" )
            out_trueflow_crops = out.get_data( larcv.kProductImage2D, "croptrueflow" )
            for iimg in xrange(ev_trueflow_crops.Image2DArray().size()):
                out_trueflow_crops.Append( ev_trueflow_crops.Image2DArray().at(iimg) )
            print("saved ",out_trueflow_crops.Image2DArray().size()," true flow crops")

        # save stitched output
        if run_stitcher:
            out_y2u = out.get_data( larcv.kProductImage2D, "larflowy2u" )
            out_y2u.Append( stitcher._outimg_v.at(0) )
            out_y2v = out.get_data( larcv.kProductImage2D, "larflowy2v" )
            out_y2v.Append( stitcher._outimg_v.at(1) )

        # save larflow hits
        if run_reco_flowhits:
            ev_larflowhits = out_ll.get_data(larlite.data.kLArFlow3DHit, "flowhits")
            for ihit in xrange(larflowhits_v.size()):
                ev_larflowhits.push_back( larflowhits_v.at(ihit) )
        if has_mc and run_truth_flowhits:
            ev_trueflowhits = out_ll.get_data(larlite.data.kLArFlow3DHit, "trueflowhits")
            for ihit in xrange(trueflowhits_v.size()):
                ev_trueflowhits.push_back( trueflowhits_v.at(ihit) )

        # set id
        out.set_id( run, subrun, event )
        out_ll.set_id( run, subrun, event )

        # save entry
        out.save_entry()
        out_ll.next_event()

        dt_result = time.time()-tresult

        # clear processor iomanager of  the entry
        io.clear_entry()
        nevents += 1


    dt_tot = time.time()-ttot

    print( "Total run time: %.3f secs"%(dt_tot))
    print( "  Data loading time: %.3f secs (%.3f secs/event)"%(dt_data, dt_data/nevents))
    print( "  Prepare data for net: %.3f secs (%.3f secs/image)"%(dt_aten, dt_aten/nevents))
    print( "  Net running time: %.3f secs (%.3f secs/event, %.3f secs/image)"%( dt_net,  dt_net/nevents, dt_net/nimgs ))
    print( "  FlowHits running time: %.3f secs (%.3f secs/image)"%( dt_flow,  dt_flow/nevents ))    
    print( "  Result conversion: %.3f secs (%.3f secs/image)"%( dt_result,  dt_result/nevents ))

    out.finalize()
    out_ll.close()
    splitter.finalize()

    return None
        

if __name__ == "__main__":

    from crop_processor_cfg import fullsplit_processor_config
        
    args = parser.parse_args(sys.argv[1:])
    
    inputfiles = ["testdata/larcvtruth-Run000002-SubRun002000.root"]
    weightfile = "weights/dualflow/dualflow.classvec.checkpoint.242000th.tar"

    inputfiles = args.inputfiles
    weightfile = args.weight_dir+"/"+args.weight_file
    if not os.path.exists(weightfile):
        raise ValueError("Cannot find weight file: {}".format(weightfile))
    output_larcv_sparsecrops   = args.outfile_larcv
    output_larlite_larflowhits = args.outfile_larlite
    if not args.overwrite and os.path.exists(args.outfile_larcv):
        raise ValueError("LArCV output already exists: {}".format(args.outfile_larcv))
    if not args.overwrite and os.path.exists(args.outfile_larlite):
        raise ValueError("Larlite output already exists: {}".format(args.outfile_larlite))

    run_reco_flowhits = True
    run_truth_flowhits = True
    if args.no_flowhits:
        run_reco_flowhits  = False
        run_truth_flowhits = False
    
    deploy_sparselarflow_on_files( output_larcv_sparsecrops, output_larlite_larflowhits,
                                   inputfiles, weightfile,
                                   model_name="dualflow_classvec_v2",
                                   devicename="cpu",
                                   adc_producer=args.adc_producer,
                                   run_reco_flowhits=run_reco_flowhits,
                                   run_truth_flowhits=run_truth_flowhits,
                                   save_cropped_trueflow=args.save_trueflow_crops,
                                   chstatus_producer=args.chstatus_producer,
                                   maxentries=args.num, has_mc=args.has_mc )


                                 
