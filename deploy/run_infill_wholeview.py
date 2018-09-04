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
# also, implicitly loads dependencies, pytorch infill model definition (ubresnet)
from infill_funcs import load_model

class WholeImageLoader:
    def __init__(self,larcv_input_file, ismc=True):
        """ This class prepares the data.  
        It passes each event through ubsplitdet to make subimages.
        The bbox and adcs from these images are passed to ubcropinfill to crop the truth images.
        
        """
        self.ismc = ismc
        
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

        # cropper for infill (needed if we do not restitch the output)
        infillcrop_cfg_str="""Verbosity:0
        InputBBoxProducer: \"detsplit\"
        InputWireProducer: \"wire\"
        InputLabelsProducer: \"Labels\"
        InputADCProducer: \"detsplit\"
        OutputCroppedWireProducer: \"wire\"
        OutputCroppedLabelsProducer: \"Labels\"
        OutputCroppedADCProducer: \"ADC\"
        OutputCroppedWeightsProducer: \"Weights\"
        OutputCroppedMetaProducer: \"meta\"
        OutputFilename: \"baka_cropinfill.root\"
        CheckFlow: false
        MakeCheckImage: false
        DoMaxPool: false
        RowDownsampleFactor: 2
        ColDownsampleFactor: 2
        MaxImages: 10
        LimitOverlap: false
        MaxOverlapFraction: -1
        """
        infillcrop_cfg = open("infillcrop.cfg",'w')
        print >>infillcrop_cfg,infillcrop_cfg_str
        infillcrop_cfg.close()
        infillcrop_pset = larcv.CreatePSetFromFile( "infillcrop.cfg", "UBCropInfill" )
        self.infillcrop_algo = larcv.UBCropInfill()
        self.infillcrop_algo.configure( infillcrop_pset )
        self.infillcrop_algo.initialize()
        self.infillcrop_algo.set_verbosity(0)
        
        self._nentries = self.io.get_n_entries()
        

    def nentries(self):
        return self._nentries

    def get_split_entry( self, entry ):
        self.io.read_entry(entry)
        # first split the entry
        self.split_algo.process( self.io )
        ev_split_bbox = self.io.get_data("bbox2d","detsplit")
        ev_adc_crops  = self.io.get_data("image2d","detsplit")
        return {"bbox":ev_split_bbox,"ADC":ev_adc_crops}

    def get_infill_cropped(self):
        if self.ismc:
            print "run infill cropper"
            self.infillcrop_algo.process( self.io )            
            ev_labels_crops  = self.io.get_data("image2d","Labels")
            ev_weights_crops = self.io.get_data("image2d","Weights")
            print "retrieve infill cropped images: ",ev_labels_cros.as_vector().size()
            data = {"labels":ev_labels_crops,"weights":ev_weights_crops}
        else:
            data = {}
        return data

if __name__=="__main__":

    # ARGUMENTS DEFINTION/PARSER
    if len(sys.argv)>1:
        whole_view_parser = argparse.ArgumentParser(description='Process whole-image views through Infill.')
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
        output_larcv_filename = "larcv_infill_5482426_95_testsample082918.root"
        # bnbmc+overlay
        #input_larcv_filename = "../testdata/supera-Run006999-SubRun000013-overlay.root"
        #output_larcv_filename = "larcv_infill_overlay_6999_13.root"
        checkpoint_data = ["../weights/dev_filtered/devfiltered_infill_final_checkpoint_uplane.tar",
                           "../weights/dev_filtered/devfiltered_infill_final_checkpoint_vplane.tar",
                           "../weights/dev_filtered/devfiltered_infill_final_checkpoint_yplane.tar"]
        batch_size = 1
        gpuid = 0
        checkpoint_gpuid = 0
        verbose = False
        nprocess_events = -1
        stitch = False
        ismc = False # saves flow and visi images
        save_cropped_adc = True # remove for y2v so we can hadd with y2u output
        use_half = True

    # load data
    inputdata = WholeImageLoader( input_larcv_filename, ismc=ismc )
    
    # load models (all nets if possible)
    models = [None,None,None]
    for p in xrange(3):
        models[p] = load_model( checkpoint_data[p], gpuid=gpuid, checkpointgpu=checkpoint_gpuid, use_half=use_half )
        models[p].to(device=torch.device("cuda:%d"%(gpuid)))
        models[p].eval()
    print "MODELS LOADED"

    # output IOManager
    if stitch:
        outputdata = larcv.IOManager( larcv.IOManager.kBOTH )        
        outputdata.add_in_file(  input_larcv_filename )
    else:
        # if not stiching we will save crops of adc,flow, and visi
        outputdata = larcv.IOManager( larcv.IOManager.kWRITE )
    outputdata.set_out_file( output_larcv_filename )
    outputdata.initialize()

    # Infill subimage stitcher
    if stitch:
        #stitcher = larcv.UBInfillStitcher("flow")
        raise RuntimeError("Not implemented yet")
    else:
        stitcher = None

    timing = OrderedDict()
    timing["total"]              = 0.0
    timing["+entry"]             = 0.0
    timing["++load_larcv_data:ubsplitdet"]  = 0.0
    timing["++load_larcv_data:ubcropinfill"]  = 0.0
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
        split_data = inputdata.get_split_entry(ientry)
        splitimg_bbox_v = split_data["bbox"]
        splitimg_adc_v  = split_data["ADC"].as_vector()
        nimgs = splitimg_bbox_v.size() 
        tdata = time.time()-tdata
        timing["++load_larcv_data:ubsplitdet"] += tdata
        tdata = time.time()
        if stitch:
            infill_cropped_dict = None
        else:
            infill_cropped_dict = inputdata.get_infill_cropped()
            print "Infill Cropper Produced: "
            print "  adc: ",split_data["ADC"].as_vector().size()
            if ismc:
                print "  Labels: ",infill_cropped_dict["Labels"].as_vector().size()
                print "  Weights: ",infill_cropped_dict["Weights"].as_vector().size()
            else:
                print "  Labels: None-not MC"
                print "  Weights: None-not MC"
        tdata = time.time()-tdata
        timing["++load_larcv_data:ubcropinfill"] += tdata
        if verbose:
            print "time to get images: ",tdata," secs"
        
        if verbose:
            print "number of images in whole-view split: ",nimgs


        # get input adc images (wholeview)
        talloc = time.time()
        ev_img = inputdata.io.get_data("image2d","wire")
        img_v = ev_img.as_vector()
        img_np = np.zeros( (img_v.size(),1,img_v.front().meta().rows(),img_v.front().meta().cols()), dtype=np.float32 )
        orig_meta = [ img_v[x].meta() for x in range(3) ]
        runid    = ev_img.run()
        subrunid = ev_img.subrun()
        eventid  = ev_img.event()
        print "Whole-view ADC Input image. Nimgs=",img_v.size()," (rse)=",(runid,subrunid,eventid)

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

        # allocate array for input adc (each plane)
        source_np = [ np.zeros( (batch_size,1,512,832), dtype=np.float32 ) for x in xrange(0,3) ]
        result_np = [ np.zeros( (batch_size,1,512,832), dtype=np.float32 ) for x in xrange(0,3) ]        

        talloc = time.time()-talloc
        timing["++alloc_arrays"] += talloc
        if verbose:
            print "time to allocate memory (and copy) for numpy arrays: ",talloc,"secs"

        nsets = nimgs/3

        # loop over images from cropper+each plane
        iset   = 0 # index of image in cropper
        ibatch = 0 # current batch index, once it hits batch size (or we run out of images, we run the network)
        while iset<nsets:

            # each set is (u,v,y) 
            
            if verbose:
                print "starting at iimg=",iimg," set=",iset
            tformat = time.time() # time to get info into torch format


            # ----------------------------------
            # Batch Prep, gather larcv data

            # clear batch            
            for p in xrange(0,3):
                source_np[p][:] = 0.0
                result_np[p][:] = 0.0
                
            # save meta information for the batch
            image_meta = {0:[],1:[],2:[]}
            infill_batch = [] # holds larcv data for batch
            for ib in range(batch_size):
                # set index of first U-plane image in the cropper set
                iimg = 3*iset

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
                    for p in xrange(3):
                        image_meta[p].append(None)
                    # skip to next item in batch
                    continue

                # pass image data to numpy array and store meta
                for p in xrange(0,3):
                    adcimg = splitimg_adc_v.at(iimg+p)
                    source_np[p][ib,0,:] = larcv.as_ndarray( adcimg )
                    image_meta[p].append( splitimg_adc_v.at(iimg+p).meta()  )

                # if not stiching, save crops
                if not stitch:
                    infillcrops = {"adc":[],"weights":[],"labels":[]}
                    for p in xrange(0,3):
                        infillcrops["adc"].append(  splitimg_adc_v.at( iimg+p )  )
                        if ismc:
                            infillcrops["weights"].append( infill_cropped_dict["weights"].at( iimg+p ) )
                            infillcrops["labels"].append( infill_cropped_dict["labels"].at( iimg+p ) )
                    infill_batch.append( infillcrops )

                iset += 1
                if iset>=nsets:
                    # then break loop
                    break
                    
            # end of batch larcv prep
            # ------------------------

            if verbose:
                print "batch using ",len(image_meta)," slots"

            # --------------------------
            # Run batch through network, for each plane
            for p in xrange(0,p):
                
                # filled batch, make tensors
                source_t = torch.from_numpy( source_np[p] ).to(device=torch.device("cuda:%d"%(gpuid)))
                if use_half:
                    source_t = source_t.half()
                tformat = time.time()-tformat
                timing["+++format"] += tformat
                if verbose:
                    print "time to slice and fill tensor batch (one plane): ",tformat," secs"

                # run model
                trun = time.time()
                pred_infill = models[p].forward( source_t )
                torch.cuda.synchronize() # to give accurate time use
                trun = time.time()-trun
                timing["+++run_model"] += trun
                if verbose:
                    print "time to run model (one plane): ",trun," secs"            

                # get result tensor
                result_np[p] = pred_infill.detach().cpu().numpy().astype(np.float32)

            # -------------
            # Store results
            tcopy = time.time()            
            for ib in xrange(min(batch_size,len(image_meta[p]))):
                isgood = True
                for p in xrange(3):
                    if image_meta[p][ib] is None:
                        isgood = False
                if not isgood:
                    continue

                # convert data to larcv
                infill_lcv = [ larcv.as_image2d_meta( result_np[p][ib,0,:], image_meta[p][ib] ) for p in xrange(3) ]

                # if stiching, store into stitch
                if stitch:
                    outmeta = out_v[p].meta() # stitch meta                    
                    for p in xrange(3):
                        stitcher.insertFlowSubimage( infill_lcv[p], image_meta[p][ib] )

                # we save flow image and crops for each prediction
                if not stitch:
                    evoutpred = outputdata.get_data("image2d","infillCropped")
                    for lcv in infill_lcv:
                        evoutpred.append( lcv )
                    if save_cropped_adc:
                        evoutadc  = outputdata.get_data("image2d","adcCropped")
                        for img in infill_batch[ib]["adc"]:
                            evoutadc.append( img )
                    
                    if ismc:
                        #evoutvisi = outputdata.get_data("image2d","pixvisi")
                        #evoutflow = outputdata.get_data("image2d","pixflow")                                            
                        #for img in flowcrop_batch[ib]["visi"]:
                        #    evoutvisi.append( img )
                        #for img in flowcrop_batch[ib]["flow"]:
                        #    evoutflow.append( img )
                        pass
                    
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

    


