#!/usr/bin/env python
# builtins
import os,sys,time
from collections import OrderedDict
import argparse

# ----------------------------------------------------------------------
# PARSE ARGUMENTS 

whole_view_parser = argparse.ArgumentParser(description='Process whole-image views through LArFlow.')
whole_view_parser.add_argument( "-i", "--input",        required=True, type=str, help="location of input larcv file" )
whole_view_parser.add_argument( "-o", "--output",       required=True, type=str, help="location of output larcv file" )
whole_view_parser.add_argument( "-c", "--checkpoint",   required=True, type=str, help="location of model checkpoint file")
whole_view_parser.add_argument( "-f", "--flowdir",      required=True, type=str, help="Flow direction. Choose from [Y2U,Y2V]")
whole_view_parser.add_argument( "-g", "--gpuid",        default=0,     type=int, help="GPUID to run on. If ID<0 then use CPU.")
whole_view_parser.add_argument( "-p", "--chkpt-gpuid",  default=0,     type=int, help="GPUID used in checkpoint")
whole_view_parser.add_argument( "-b", "--batchsize",    default=1,     type=int, help="batch size" )
whole_view_parser.add_argument( "-v", "--verbose",      action="store_true",     help="verbose output")
whole_view_parser.add_argument( "-n", "--nevents",      default=-1,    type=int, help="process number of events (-1=all)")
whole_view_parser.add_argument( "-t", "--adc-threshold",default=5.0,   type=float, help="process number of events (-1=all)")
whole_view_parser.add_argument( "-s", "--stitch",       action="store_true", default=False, help="stitch info from cropped images into whole view again. else save cropped info." )
whole_view_parser.add_argument( "-mc","--ismc",         action="store_true", default=False, help="use flag if input file is MC or not" )
whole_view_parser.add_argument( "-hp","--usehalf",      action="store_true", default=False, help="use half-precision values" )
whole_view_parser.add_argument( "-d", "--debug",        action="store_true", default=False, help="run in debug mode. uses hardcoded parameters for dev" )
whole_view_parser.add_argument( "-a", "--saveadc",      action="store_true", default=False, help="save cropped ADC as well" )
whole_view_parser.add_argument( "-w", "--workdir",      default="./", type=str, help="set working directory" )
whole_view_parser.add_argument( "-adc", "--adc-producer",     default="wire", type=str, help="Wholeview ADC Image producer name" )
whole_view_parser.add_argument( "-ch",  "--chstatus-producer",default="wire", type=str, help="Channel Status producer name" )                                

args = whole_view_parser.parse_args(sys.argv[1:])

# -------------------------------------------------------------------



# numpy
import numpy as np

# ROOT/larcv
import ROOT as rt
from larcv import larcv
from ublarcvapp import ublarcvapp

# pytorch
import torch

# util functions
# also, implicitly loads dependencies, pytorch larflow model definition
from larflow_funcs import load_model

# cv2: opencv2 for dumping input numpy arrays
try:
    import cv2
    _HAS_CV2_ = True
except:
    _HAS_CV2_ = False

class WholeImageLoader:
    def __init__(self,larcv_input_file,
                 adc_producer="wire", chstatus_producer="wire",
                 tick_backward=True,
                 ismc=True,workdir="./"):

        # we setup a larcv IOManager for read mode
        tick_direction = larcv.IOManager.kTickForward
        if tick_backward:
            tick_direction = larcv.IOManager.kTickBackward
        self.io = larcv.IOManager( larcv.IOManager.kBOTH, "input", tick_direction )
        self.io.add_in_file( larcv_input_file )
        self.io.set_out_file( "temp_deploy_splitter_file.root" )
        self.io.initialize()

        # we setup some image processor modules

        # split a whole image into 3D-consistent chunks
        # the module will return bounding box defintions
        # the event loop will do the slicing
        ubsplit_cfg="""
        InputProducer: \"%s\"
        OutputBBox2DProducer: \"detsplit\"
        CropInModule: true
        OutputCroppedProducer: \"detsplit\"
        BBoxPixelHeight: 512
        BBoxPixelWidth: 832
        CoveredZWidth: 310
        FillCroppedYImageCompletely: true
        DebugImage: false
        MaxImages: -1
        RandomizeCrops: false
        MaxRandomAttempts: 1000
        MinFracPixelsInCrop: 0.0
        """%(adc_producer)
        fcfg = open(workdir+"/ubsplit.cfg",'w')
        print >>fcfg,ubsplit_cfg
        fcfg.close()
        split_pset = larcv.CreatePSetFromFile( workdir+"./ubsplit.cfg", "UBSplitDetector" )
        self.split_algo = ublarcvapp.UBSplitDetector()
        self.split_algo.configure(split_pset)
        self.split_algo.initialize()
        self.split_algo.set_verbosity(0)

        # cropper for larflow (needed if we do not restitch the output)
        lfcrop_cfg="""Verbosity:0
        InputBBoxProducer: \"detsplit\"
        InputCroppedADCProducer: \"detsplit\"
        InputADCProducer: \"{}\"
        InputChStatusProducer: \"{}\"
        InputVisiProducer: \"pixvisi\"
        InputFlowProducer: \"pixflow\"
        OutputCroppedADCProducer:  \"adc\"
        OutputCroppedVisiProducer: \"visi\"
        OutputCroppedFlowProducer: \"flow\"
        OutputCroppedMetaProducer: \"flowmeta\"
        OutputFilename: \"baka_lf.root\"
        SaveOutput: false
        CheckFlow:  false
        MakeCheckImage: false
        DoMaxPool: false
        RowDownsampleFactor: 2
        ColDownsampleFactor: 2
        MaxImages: -1
        LimitOverlap: false
        RequireMinGoodPixels: false
        MaxOverlapFraction: 0.2
        UseVectorizedCode: true
        HasVisibilityImage: false
        SaveTrainingOutput: false
        IsMC: {}
        """
        flowcrop_cfg = open(workdir+"/ublarflowcrop.cfg",'w')
        print >>flowcrop_cfg,lfcrop_cfg.format( adc_producer,
                                                chstatus_producer,
                                                str(ismc).lower() )
        flowcrop_cfg.close()
        flowcrop_pset = larcv.CreatePSetFromFile( workdir+"/ublarflowcrop.cfg", "UBLArFlowCrop" )
        self.flowcrop_algo = ublarcvapp.UBCropLArFlow()
        self.flowcrop_algo.configure( flowcrop_pset )
        self.flowcrop_algo.initialize()
        self.flowcrop_algo.set_verbosity(0)
        self.ismc = ismc
        
        self._nentries = self.io.get_n_entries()
        

    def nentries(self):
        return self._nentries

    def get_entry( self, entry ):
        self.io.read_entry(entry)
        self.split_algo.process( self.io )

        #ev_split_imgs = self.io.get_data("image2d","detsplit")
        #return ev_split_imgs.image2d_array()        
        ev_split_bbox = self.io.get_data(larcv.kProductROI,"detsplit")
        return ev_split_bbox

    def get_larflow_cropped(self):
        print "run larflow cropper"
        ev_adc_crops  = self.io.get_data(larcv.kProductImage2D,"detsplit")
        print "retrieve larflow cropped images: ",ev_adc_crops.Image2DArray().size()
        if self.ismc:
            self.flowcrop_algo.process( self.io )            
            ev_flow_crops = self.io.get_data(larcv.kProductImage2D,"flow")
            ev_visi_crops = self.io.get_data(larcv.kProductImage2D,"visi")
            data = {"adc":ev_adc_crops,"flow":ev_flow_crops,"visi":ev_visi_crops}
        else:
            data = {"adc":ev_adc_crops,"flow":None,"visi":None}
        return data

if __name__=="__main__":

    # ARGUMENTS DEFINTION/PARSER
    if not args.debug:
        # required
        input_larcv_filename  = args.input
        output_larcv_filename = args.output
        checkpoint_data       = args.checkpoint
        FLOWDIR               = args.flowdir.upper()

        # has defaults
        batch_size            = args.batchsize        
        gpuid                 = args.gpuid
        if gpuid<0:
            gpuid = "cpu"
        checkpoint_gpuid      = args.chkpt_gpuid
        verbose               = args.verbose        
        nprocess_events       = args.nevents
        stitch                = args.stitch
        use_half              = args.usehalf
        if gpuid=="cpu":
            use_half = False # no half-precision operations for CPU
        ismc                  = args.ismc
        save_cropped_adc      = args.saveadc
        workdir               = args.workdir
        threshold             = args.adc_threshold
        adc_producer          = args.adc_producer
        chstatus_producer     = args.chstatus_producer
        
    else:
        print "OVER-RIDING ARGUMENTS: USING DEBUG SETTINGS (Hard-coded)"
        # for testing
        input_larcv_filename = "../../testdata/mcc9_v13_bnbnue_corsika/larcv_mctruth_000de2ae-28d2-46c3-9dc9-af1fee95eaa5.root"
        batch_size = 4
        #gpuid = "cpu"
        gpuid = 0        
        checkpoint_gpuid = 0
        verbose = True
        nprocess_events = 1
        stitch   = False
        use_half = False
        workdir="./"
        threshold = 10.0
        adc_producer="wiremc"
        chstatus_producer="wiremc"
        ismc = False # saves flow and visi images
        
        FLOWDIR="Y2U"

        if FLOWDIR=="Y2U":
            checkpoint_data = "weights/dev_filtered/devfiltered_larflow_y2u_832x512_32inplanes.tar"
            output_larcv_filename = "larcv_larflow_y2u_debug.root"            
            save_cropped_adc = True  # saves cropped adc
        elif FLOWDIR=="Y2V":
            checkpoint_data = "weights/dev_filtered/devfiltered_larflow_y2v_832x512_32inplanes.tar"
            output_larcv_filename = "larcv_larflow_y2v_debug.root"
            # remove for y2v so we can hadd with y2u output                        
            ismc = False
            save_cropped_adc = False 
        else:
            raise RuntimeError("invalid flowdir")


    if FLOWDIR not in ["Y2U","Y2V"]:
        raise ValueError("Required argument '--flowdir' must be [ Y2U, Y2V ]")

    if type(gpuid) is int:
        device=torch.device("cuda:%d"%(gpuid))
    elif gpuid=="cpu":
        device=torch.device("cpu")
    print "DEVICE: ",device
    
    # load data
    inputdata = WholeImageLoader( input_larcv_filename, ismc=ismc, workdir=workdir,
                                  adc_producer=adc_producer, chstatus_producer=chstatus_producer )
    
    # load model
    model = load_model( checkpoint_data, deviceid=gpuid,
                        checkpointgpu=checkpoint_gpuid, use_half=use_half )
    model.to(device=device)
    model.eval()

    # set planes
    if FLOWDIR=="Y2U":
        target_plane = 0
    elif FLOWDIR=="Y2V":
        target_plane = 1
    else:
        raise ValueError("invalid FLOWDIR value")

    # output IOManager
    if stitch:
        outputdata = larcv.IOManager( larcv.IOManager.kBOTH )        
        outputdata.add_in_file(  input_larcv_filename )
    else:
        # if not stiching we will save crops of adc,flow, and visi
        outputdata = larcv.IOManager( larcv.IOManager.kWRITE )
    outputdata.set_out_file( output_larcv_filename )
    outputdata.initialize()

    # LArFlow subimage stitcher
    if stitch:
        stitcher = larcv.UBLArFlowStitcher("flow")
    else:
        stitcher = None

    timing = OrderedDict()
    timing["total"]              = 0.0
    timing["+entry"]             = 0.0
    timing["++load_larcv_data:ubsplitdet"]  = 0.0
    timing["++load_larcv_data:ubcroplarflow"]  = 0.0
    timing["++alloc_arrays"]     = 0.0
    timing["+++format"]          = 0.0
    timing["+++run_model"]       = 0.0
    timing["+++copy_to_output"]  = 0.0
    timing["++save_output"]      = 0.0

    ttotal = time.time()

    nevts = inputdata.nentries()
    if nprocess_events>=0:
        nevts = nprocess_events

    for ientry in range(nevts):

        if verbose:
            print "=== [ENTRY %d] ==="%(ientry)
        
        tentry = time.time()
        
        # get the input data: data loader reads in whole view and splits and crops
        tdata = time.time()
        splitimg_bbox_v = inputdata.get_entry(ientry)
        nimgs = splitimg_bbox_v.ROIArray().size() 
        tdata = time.time()-tdata
        timing["++load_larcv_data:ubsplitdet"] += tdata
        tdata = time.time()
        if stitch:
            larflow_cropped_dict = None
        else:
            larflow_cropped_dict = inputdata.get_larflow_cropped()
            #print "LArFlow Cropper Produced: "
            #print "  adc: ",larflow_cropped_dict["adc"].image2d_array().size()
            #if ismc:
            #    print "  visi: ",larflow_cropped_dict["visi"].image2d_array().size()
            #    print "  flow: ",larflow_cropped_dict["flow"].image2d_array().size()
            #else:
            #    print "  visi: None-not MC"
            #    print "  flow: None-not MC"
        tdata = time.time()-tdata
        timing["++load_larcv_data:ubcroplarflow"] += tdata
        if verbose:
            print "time to get images: ",tdata," secs"
        
        if verbose:
            print "number of images in whole-view split: ",nimgs


        # get input wholeview adc images
        talloc = time.time()
        ev_img = inputdata.io.get_data(larcv.kProductImage2D,adc_producer)
        img_v  = ev_img.Image2DArray()
        img_np = np.zeros( (img_v.size(),1,img_v.front().meta().rows(),img_v.front().meta().cols()), dtype=np.float32 )
        orig_meta = [ img_v[x].meta() for x in range(3) ]
        runid    = ev_img.run()
        subrunid = ev_img.subrun()
        eventid  = ev_img.event()
        print "ADC Input image. Nimgs=",img_v.size()," (rse)=",(runid,subrunid,eventid)

        # get chstatus, which we will use to zero out info
        ev_chstatus = inputdata.io.get_data(larcv.kProductChStatus, chstatus_producer)
        chstatus_np = larcv.as_ndarray( ev_chstatus )

        # setup stitcher
        if stitch:
            stitcher.setupEvent( img_v )
        
        # output stitched images
        out_v = rt.std.vector("larcv::Image2D")()
        for i in range(img_v.size()):
            img_np[i,0,:,:] = np.transpose( larcv.as_ndarray( img_v[i] ), (1,0) )
            out = larcv.Image2D( img_v[i].meta() )
            out.paint(0.0)
            out_v.push_back( out )

        # fill source and target images
        source_np = np.zeros( (batch_size,1,512,832), dtype=np.float32 )
        target_np = np.zeros( (batch_size,1,512,832), dtype=np.float32 )

        talloc = time.time()-talloc
        timing["++alloc_arrays"] += talloc
        if verbose:
            print "time to allocate memory (and copy) for numpy arrays: ",talloc,"secs"

        nsets = nimgs

        # loop over images from cropper
        iset   = 0 # index of image in cropper
        ibatch = 0 # current batch index, once it hits batch size (or we run out of images, we run the network)
        while iset<nsets:
            
            if verbose:
                print "starting set=",iset," of ",nsets
            tformat = time.time() # time to get info into torch format

            # -------------------------------------------------
            # Batch Loop, fill data, then send through the net
            # clear batch
            source_np[:] = 0.0
            target_np[:] = 0.0
            idx_target_flow = {"Y2U":0, "Y2V":1}            
            
            # save meta information for the batch
            image_meta = []
            target_meta = []
            flowcrop_batch = [] # only filled if not stitching
            status_batch = []
            for ib in range(batch_size):
                # set index of first U-plane image in the cropper set
                iimg = 3*iset
                if verbose:
                    print "iimg=",iimg," of nimgs=",nimgs," of nbboxes=",splitimg_bbox_v.ROIArray().size()
                # get the bboxes, all three planes
                bb_v  = [splitimg_bbox_v.ROIArray().at(iset).BB(x) for x in xrange(3)]

                if False:
                    # do crop oneself using numpy
                    bounds = []
                    isbad = False
                    # get row,col bounds for each plane
                    for bb,orig in zip(bb_v,orig_meta):
                        rmin = orig.row( bb.min_y() )
                        rmax = orig.row( bb.max_y() )
                        cmin = orig.col( bb.min_x() )
                        cmax = orig.col( bb.max_x() )
                        if rmax-rmin!=512 or cmax-cmin!=832:
                            print "[ERROR] about bbox size: (rmin,rmax,cmin,cmax)=",rmin,rmax,cmin,cmax
                            print "  image metas: "
                            for ip,bbb in enumerate(bb_v):
                                print "  plane ",ip,": ",bbb.dump()
                            isbad = True
                        bounds.append( (rmin,cmin,rmax,cmax) )
                    
                    # we have to get the row, col bounds in the source image
                    if isbad:
                        #sign of bad image
                        image_meta.append(None) 
                        target_meta.append(None)
                        status_batch.append(None)
                        continue

                # crops in numpy array
                src_img_lcv = larflow_cropped_dict["adc"].Image2DArray().at( iimg+2 )
                tar_img_lcv = larflow_cropped_dict["adc"].Image2DArray().at( iimg+idx_target_flow[FLOWDIR] )
                source_np[ib,0,:,:] = np.transpose( larcv.as_ndarray(src_img_lcv), (1,0) )
                target_np[ib,0,:,:] = np.transpose( larcv.as_ndarray(tar_img_lcv), (1,0) )
                #status_batch.append( chstatus_np[2,bounds[2][1]:bounds[2][3]] )

                if args.debug:
                    cv2.imwrite( "debug_src_img_set%d.png"%(iset), source_np[ib,0,:,:] )
                
                # store region of image                
                image_meta.append(  src_img_lcv.meta() )
                target_meta.append( tar_img_lcv.meta() )

                # if not stiching, save crops, we might be saving these
                if not stitch:
                    flowcrops = {"flow":[],"visi":[],"adc":[]}
                    for ii in xrange(0,3):
                        flowcrops["adc"].append(  larflow_cropped_dict["adc"].at( iimg+ii )  )
                    if ismc:
                        for ii in xrange(0,2):
                            flowcrops["flow"].append( larflow_cropped_dict["flow"].at( iset*2+ii ) )
                            flowcrops["visi"].append( larflow_cropped_dict["visi"].at( iset*2+ii ) )

                    flowcrop_batch.append( flowcrops )

                iset += 1
                if iset>=nsets:
                    # then break loop
                    break
                    
            # end of batch prep loop
            # -----------------------            

            if verbose:
                print "batch using ",len(image_meta)," of ",batch_size,"slots"
        
            # filled batch, make tensors
            # --------------------------
            source_t = torch.from_numpy( source_np ).to(device=device)
            target_t = torch.from_numpy( target_np ).to(device=device)
            if use_half:
                source_t = source_t.half()
                target_t = target_t.half()
            tformat = time.time()-tformat
            timing["+++format"] += tformat
            if verbose:
                print "time to slice and fill tensor batch: ",tformat," secs"

            # run model
            trun = time.time()
            with torch.set_grad_enabled(False):
                pred_flow, pred_visi = model.forward( source_t, target_t )
            if gpuid>=0:
                torch.cuda.synchronize() # to give accurate time use
            trun = time.time()-trun
            timing["+++run_model"] += trun
            if verbose:            
                print "time to run model: ",trun," secs"            

            # turn pred_flow back into larcv
            tcopy = time.time()
            if verbose:
                print "prediction to cpu"
            flow_np = pred_flow.detach().cpu().numpy().astype(np.float32)

            
            outmeta = out_v[0].meta()
            for ib in xrange(min(batch_size,len(image_meta))):
                if image_meta[ib] is None:
                    # bad image for whatever reason
                    continue
                flow_slice = flow_np[ib,0,:,:]
                #print "flow_slice non-zero (pre-mask): ",(flow_slice!=0).sum()
                flow_slice[ source_np[ib,0,:,:]<threshold ] = 0.0
                
                if args.debug:
                    cv2.imwrite( "debug_flowout_{}_set{}_batchidx{}.png".format( FLOWDIR, iset, ib ), flow_slice )
                

                # we want to suppress values where chstatus is good and adc value below threshold
                # setting uninteresting pixels to zero is important for good file size
                thresh_slice = (source_np[ib,0,:,:]<threshold)
                #adcgoodch_slice = source_np[ib,0,:,status_batch[ib]==4].transpose((1,0))
                
                # SKIP CHSTATUS MASK FOR NOW
                #nmasked = 0
                #for n,goodch in enumerate( np.nditer( status_batch[ib] ) ):
                #    if goodch==4:
                #        nmasked += (flow_slice[:,n]<threshold).sum()
                #        flow_slice[:,n][ source_np[ib,0,:,n]<threshold ] = 0

                # zero regions in good channel list and below threshold
                #print "nmasked estimate: ",nmasked
                #print "flow_slice non-zero (post-mask): ",(flow_slice!=0).sum()

                flow_lcv = larcv.as_image2d_meta( np.transpose( flow_slice, (1,0 ) ), image_meta[ib] )
                if stitch:
                    # load cropped info into stitcher
                    # -------------------------------
                    stitcher.insertFlowSubimage( flow_lcv, target_meta[ib] )
                else:
                    # save cropped images
                    #--------------------
                    # prediction images
                    evoutpred = outputdata.get_data(larcv.kProductImage2D,"larflow_%s"%(FLOWDIR.lower()))
                    evoutpred.Append( flow_lcv )

                    # input cropped source and target image
                    if save_cropped_adc:
                        evoutadc  = outputdata.get_data(larcv.kProductImage2D,"adc")
                        for img in flowcrop_batch[ib]["adc"]:
                            evoutadc.Append( img )
                    # save truth flow and visi crops
                    if ismc:
                        evoutvisi = outputdata.get_data(larcv.kProductImage2D,"pixvisi")
                        evoutflow = outputdata.get_data(larcv.kProductImage2D,"pixflow")                                            
                        for img in flowcrop_batch[ib]["visi"]:
                            evoutvisi.Append( img )
                        for img in flowcrop_batch[ib]["flow"]:
                            evoutflow.Append( img )

            tcopy = time.time()-tcopy
            timing["+++copy_to_output"] += tcopy
            if verbose:
                print "time to copy results back into full image: ",tcopy," secs"
        # end of loop over cropped set
        # -------------------------------

        if verbose:
            print "Save Entry"
            
        outputdata.set_id( runid, subrunid, eventid )
        outputdata.save_entry()
        

        # end of while loop
        if verbose:
            print "Processed all the images"

        tout = time.time()
        if stitch:
            outputdata.read_entry(ientry)
            stitcher.process( outputdata )
            outputdata.save_entry()
        else:
            pass
        tout = time.time()-tout
        timing["++save_output"] += tout

        tentry = time.time()-tentry
        if verbose:
            print "time for entry: ",tentry,"secs"
        timing["+entry"] += tentry

    # save results
    outputdata.finalize()

    print "DONE."
    
    ttotal = time.time()-ttotal
    timing["total"] = ttotal

    print "------ TIMING ---------"
    for k,v in timing.items():
        print k,": ",v," (per event: ",v/float(nevts)," secs)"

    


