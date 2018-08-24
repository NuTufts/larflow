# builtins
import os,sys,time
from collections import OrderedDict
import argparse

# numpy
import numpy as np

# ROOT/larcv
import ROOT as rt
from larcv import larcv

# pytorch
import torch

# util functions
# also, implicitly loads dependencies, pytorch larflow model definition
from larflow_funcs import load_model

class WholeImageLoader:
    def __init__(self,larcv_input_file, ismc=True):

        # we setup a larcv IOManager for read mode
        self.io = larcv.IOManager( larcv.IOManager.kREAD )
        self.io.add_in_file( larcv_input_file )
        self.io.initialize()

        # we setup some image processor modules

        # split a whole image into 3D-consistent chunks
        # the module will return bounding box defintions
        # the event loop will do the slicing
        ubsplit_cfg="""
        InputProducer: \"wire\"
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
        """
        fcfg = open("ubsplit.cfg",'w')
        print >>fcfg,ubsplit_cfg
        fcfg.close()
        split_pset = larcv.CreatePSetFromFile( "ubsplit.cfg", "UBSplitDetector" )
        self.split_algo = larcv.UBSplitDetector()
        self.split_algo.configure(split_pset)
        self.split_algo.initialize()
        self.split_algo.set_verbosity(0)

        # cropper for larflow (needed if we do not restitch the output)
        lfcrop_cfg="""Verbosity:0
        InputBBoxProducer: \"detsplit\"
        InputCroppedADCProducer: \"detsplit\"
        InputADCProducer: \"wire\"
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
        IsMC: {}
        """
        flowcrop_cfg = open("ublarflowcrop.cfg",'w')
        print >>flowcrop_cfg,lfcrop_cfg.format( str(ismc).lower() )
        flowcrop_cfg.close()
        flowcrop_pset = larcv.CreatePSetFromFile( "ublarflowcrop.cfg", "UBLArFlowCrop" )
        self.flowcrop_algo = larcv.UBCropLArFlow()
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
        ev_split_bbox = self.io.get_data("bbox2d","detsplit")
        return ev_split_bbox

    def get_larflow_cropped(self):
        print "run larflow cropper"
        self.flowcrop_algo.process( self.io )
        ev_adc_crops  = self.io.get_data("image2d","adc")
        print "retrieve larflow cropped images: ",ev_adc_crops.image2d_array().size()
        if self.ismc:
            ev_flow_crops = self.io.get_data("image2d","flow")
            ev_visi_crops = self.io.get_data("image2d","visi")
            data = {"adc":ev_adc_crops,"flow":ev_flow_crops,"visi":ev_visi_crops}
        else:
            data = {"adc":ev_adc_crops}
        return data

if __name__=="__main__":

    # ARGUMENTS DEFINTION/PARSER
    if len(sys.argv)>1:
        whole_view_parser = argparse.ArgumentParser(description='Process whole-image views through LArFlow.')
        whole_view_parser.add_argument( "-i", "--input",        required=True, type=str, help="location of input larcv file" )
        whole_view_parser.add_argument( "-o", "--output",       required=True, type=str, help="location of output larcv file" )
        whole_view_parser.add_argument( "-c", "--checkpoint",   required=True, type=str, help="location of model checkpoint file")
        whole_view_parser.add_argument( "-g", "--gpuid",        default=0,     type=int, help="GPUID to run on")
        whole_view_parser.add_argument( "-p", "--chkpt-gpuid",  default=0,     type=int, help="GPUID used in checkpoint")
        whole_view_parser.add_argument( "-b", "--batchsize",    default=2,     type=int, help="batch size" )
        whole_view_parser.add_argument( "-v", "--verbose",      action="store_true",     help="verbose output")
        whole_view_parser.add_argument( "-v", "--nevents",      default=-1,    type=int, help="process number of events (-1=all)")
        whole_view_parser.add_argument( "-s", "--stitch",       action="store_true", default=False, help="stitch info from cropped images into whole view again. else save cropped info." )
        whole_view_parser.add_argument( "-mc","--ismc",         action="store_true", default=False, help="use flag if input file is MC or not" )
        whole_view_parser.add_argument( "-h", "--usehalf",      action="store_true", default=False, help="use half-precision values" )

        args = whole_view_parser.parse_args(sys.argv)
        input_larcv_filename  = args.input
        output_larcv_filename = args.output
        checkpoint_data       = args.checkpoint
        gpuid                 = args.gpuid
        checkpoint_gpuid      = args.chkpt_gpuid
        batch_size            = args.batchsize
        verbose               = args.verbose
        nprocess_events       = args.nevents
        stitch                = args.stitch
        use_half              = args.use_half
    else:

        # for testing
        # bnb+corsicka
        input_larcv_filename = "../testdata/larcv_5482426_95.root" # whole image    
        output_larcv_filename = "larcv_larflow_smalltest_5482426_95.root"
        # bnbmc+overlay
        #input_larcv_filename = "../testdata/supera-Run006999-SubRun000013-overlay.root"
        #output_larcv_filename = "larcv_larflow_overlay_6999_13.root"
        #checkpoint_data = "../weights/dev/dev_larflow_y2u_832x512_32inplanes.tar"
        checkpoint_data = "../weights/dev_filtered/devfiltered_larflow_y2u_832x512_32inplanes.tar"
        batch_size = 1
        gpuid = 0
        checkpoint_gpuid = 0
        verbose = False
        nprocess_events = 3
        stitch = False
        ismc = True
        use_half = True

    # load data
    inputdata = WholeImageLoader( input_larcv_filename, ismc=ismc )
    
    # load model
    model = load_model( checkpoint_data, gpuid=gpuid, checkpointgpu=checkpoint_gpuid, use_half=use_half )
    model.to(device=torch.device("cuda:%d"%(gpuid)))
    model.eval()

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
        
        tdata = time.time()
        splitimg_bbox_v = inputdata.get_entry(ientry)
        nimgs = splitimg_bbox_v.size() 
        tdata = time.time()-tdata
        timing["++load_larcv_data:ubsplitdet"] += tdata
        tdata = time.time()
        if stitch:
            larflow_cropped_dict = None
        else:
            larflow_cropped_dict = inputdata.get_larflow_cropped()
            print "LArFlow Cropper Produced: "
            print "  adc: ",larflow_cropped_dict["adc"].image2d_array().size()
            if ismc:
                print "  visi: ",larflow_cropped_dict["visi"].image2d_array().size()
                print "  flow: ",larflow_cropped_dict["flow"].image2d_array().size()
            else:
                print "  visi: None-not MC"
                print "  flow: None-not MC"
        tdata = time.time()-tdata
        timing["++load_larcv_data:ubcroplarflow"] += tdata
        if verbose:
            print "time to get images: ",tdata," secs"
        
        if verbose:
            print "number of images in whole-view split: ",nimgs


        # get input adc images
        talloc = time.time()
        ev_img = inputdata.io.get_data("image2d","wire")
        img_v = ev_img.image2d_array()
        img_np = np.zeros( (img_v.size(),1,img_v.front().meta().rows(),img_v.front().meta().cols()), dtype=np.float32 )
        orig_meta = [ img_v[x].meta() for x in range(3) ]
        runid    = ev_img.run()
        subrunid = ev_img.subrun()
        eventid  = ev_img.event()
        print "ADC Input image. Nimgs=",img_v.size()," (rse)=",(runid,subrunid,eventid)

        # setup stitcher
        if stitch:
            stitcher.setupEvent( img_v )
        
        # output stitched images
        out_v = rt.std.vector("larcv::Image2D")()
        for i in range(img_v.size()):
            img_np[i,0,:,:] = larcv.as_ndarray( img_v[i] )
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

        nsets = nimgs/3

        # loop over images from cropper
        iset   = 0 # index of image in cropper
        ibatch = 0 # current batch index, once it hits batch size (or we run out of images, we run the network)
        while iset<nsets:
            
            if verbose:
                print "starting at iimg=",iimg," set=",iset
            tformat = time.time() # time to get info into torch format

            # -------------------------------------------------
            # Batch Loop, fill data, then send through the net
            # clear batch
            source_np[:] = 0.0
            target_np[:] = 0.0
            
            # save meta information for the batch
            image_meta = []
            target_meta = []
            flowcrop_batch = [] # only filled if not stitching
            for ib in range(batch_size):
                # set index of first U-plane image in the cropper set
                iimg = 3*iset 
                #print "iimg=",iimg," of nimgs=",nimgs," of nbboxes=",splitimg_bbox_v.size()
                # get the bboxes, all three planes
                bb_v  = [splitimg_bbox_v.at(iimg+x) for x in xrange(3)]
                
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
                    continue

                # crops in numpy array
                source_np[ib,0,:,:] = img_np[2,0,bounds[2][0]:bounds[2][2],bounds[2][1]:bounds[2][3]] # yplane
                target_np[ib,0,:,:] = img_np[0,0,bounds[0][0]:bounds[0][2],bounds[0][1]:bounds[0][3]] # uplane
                
                # store region of image
                image_meta.append(  larcv.ImageMeta( bb_v[2], 512, 832 ) )
                target_meta.append( larcv.ImageMeta( bb_v[0], 512, 832 ) )

                # if not stiching, save crops
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
                print "batch using ",len(image_meta)," slots"
        
            # filled batch, make tensors
            source_t = torch.from_numpy( source_np ).to(device=torch.device("cuda:%d"%(gpuid)))
            target_t = torch.from_numpy( target_np ).to(device=torch.device("cuda:%d"%(gpuid)))
            if use_half:
                source_t = source_t.half()
                target_t = target_t.half()
            tformat = time.time()-tformat
            timing["+++format"] += tformat
            if verbose:
                print "time to slice and fill tensor batch: ",tformat," secs"

            # run model
            trun = time.time()
            pred_flow, pred_visi = model.forward( source_t, target_t )
            torch.cuda.synchronize() # to give accurate time use
            trun = time.time()-trun
            timing["+++run_model"] += trun
            if verbose:
                print "time to run model: ",trun," secs"            

            # turn pred_flow back into larcv
            tcopy = time.time()
            flow_np = pred_flow.detach().cpu().numpy().astype(np.float32)
            outmeta = out_v[0].meta()
            for ib in xrange(min(batch_size,len(image_meta))):
                if image_meta[ib] is None:
                    continue
                img_slice = flow_np[ib,0,:]
                flow_lcv = larcv.as_image2d_meta( img_slice, image_meta[ib] )
                if stitch:
                    stitcher.insertFlowSubimage( flow_lcv, target_meta[ib] )
                else:
                    # we save flow image and crops for each prediction
                    evoutadc  = outputdata.get_data("image2d","adc")
                    evoutpred = outputdata.get_data("image2d","larflow_y2u")
                    for img in flowcrop_batch[ib]["adc"]:
                        evoutadc.append( img )
                    evoutpred.append( flow_lcv )
                    
                    if ismc:
                        evoutvisi = outputdata.get_data("image2d","pixvisi")
                        evoutflow = outputdata.get_data("image2d","pixflow")                                            
                        for img in flowcrop_batch[ib]["visi"]:
                            evoutvisi.append( img )
                        for img in flowcrop_batch[ib]["flow"]:
                            evoutflow.append( img )
                    
                    outputdata.set_id( runid, subrunid, eventid )
                    outputdata.save_entry()
                    
                    
            tcopy = time.time()-tcopy
            timing["+++copy_to_output"] += tcopy
            if verbose:
                print "time to copy results back into full image: ",tcopy," secs"
        # end of loop over cropped set
        # -------------------------------

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

    


