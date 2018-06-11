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
    def __init__(self,larcv_input_file):

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
        CropInModule: false
        OutputCroppedProducer: \"detsplit\"
        BBoxPixelHeight: 512
        BBoxPixelWidth: 832
        CoveredZWidth: 310
        FillCroppedYImageCompletely: true
        DebugImage: false
        MaxImages: 1000
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


#def load_pre_cropped_data( larcvdataset_configfile, batchsize=1 ):
#    """ we can just use the normal larcvdataset"""
#    iotest = LArCVDataset( larcvdataset_configfile,"ThreadProcessorTest")
#    return iotest
    

#def load_wholeimage_data( input_larcv_filename ):
#    """ if whole images are provided, we need to load the larcv processor that splits the images"""
#    return WholeImageLoader( input_larcv_filename )



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

        args = whole_view_parser.parse_args(sys.argv)
        input_larcv_filename  = args.input
        output_larcv_filename = args.output
        checkpoint_data       = args.checkpoint
        gpuid                 = args.gpuid
        checkpoint_gpuid      = args.chkpt_gpuid
        batch_size            = args.batchsize
        verbose               = args.verbose
        nprocess_events       = args.nevents        
    else:

        # for testing
        input_larcv_filename = "larcv_8537458_6.root" # whole image
        output_larcv_filename = "output_larflow.root"
        #checkpoint_data = "checkpoint_fullres_bigsample_11000th_gpu3.tar"
        checkpoint_data = "checkpoint.20000th.tar"
        batch_size = 2
        gpuid = 1
        checkpoint_gpuid = 0
        verbose = False
        nprocess_events = 10

    # load data
    inputdata = WholeImageLoader( input_larcv_filename )
    
    # load model
    model = load_model( checkpoint_data, gpuid=gpuid, checkpointgpu=checkpoint_gpuid )
    model.to(device=torch.device("cuda:%d"%(gpuid)))
    model.eval()

    # output IOManager
    outputdata = larcv.IOManager( larcv.IOManager.kBOTH )
    outputdata.add_in_file(  input_larcv_filename )
    outputdata.set_out_file( output_larcv_filename )
    outputdata.initialize()

    # LArFlow subimage stitcher
    stitcher = larcv.UBLArFlowStitcher("flow")

    timing = OrderedDict()
    timing["total"]              = 0.0
    timing["+entry"]             = 0.0
    timing["++load_larcv_data"]  = 0.0
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
        timing["++load_larcv_data"] += tdata
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

        # setup stitcher
        stitcher.setupEvent( img_v )
        
        # output images
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

        for iset in range(nsets):
            iimg = 3*iset
            if verbose:
                print "starting at iimg=",iimg," set=",iset
            tformat = time.time() # time to get info into torch format
            
            # save meta information for the batch
            image_meta = []
            target_meta = []
            for ib in range(batch_size):
                
                # get the bboxes
                bb_v  = [splitimg_bbox_v.at(iimg+x) for x in range(3)]
                
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
                
                source_np[ib,0,:,:] = img_np[2,0,bounds[2][0]:bounds[2][2],bounds[2][1]:bounds[2][3]] # yplane
                target_np[ib,0,:,:] = img_np[0,0,bounds[0][0]:bounds[0][2],bounds[0][1]:bounds[0][3]] # uplane
                # store region of image
                image_meta.append( larcv.ImageMeta( bb_v[2], 512, 832 ) )
                target_meta.append( larcv.ImageMeta( bb_v[0], 512, 832 ) )

            if verbose:
                print "batch using ",len(image_meta)," slots"
        
            # filled batch, make tensors
            source_t = torch.from_numpy( source_np ).to(device=torch.device("cuda:1"))
            target_t = torch.from_numpy( target_np ).to(device=torch.device("cuda:1"))
            tformat = time.time()-tformat
            timing["+++format"] += tformat
            if verbose:
                print "time to slice and fill tensor batch: ",tformat," secs"

            # run model
            trun = time.time()
            pred_flow, pred_visi = model.forward( source_t, target_t )
            trun = time.time()-trun
            timing["+++run_model"] += trun
            if verbose:
                print "time to run model: ",trun," secs"            

            # turn pred_flow back into larcv
            tcopy = time.time()
            flow_np = pred_flow.detach().cpu().numpy().astype(np.float32)
            outmeta = out_v[0].meta()
            for ib in range(min(batch_size,len(image_meta))):
                if image_meta[ib] is None:
                    continue
                img_slice = flow_np[ib,0,:]
                flow_lcv = larcv.as_image2d_meta( img_slice, image_meta[ib] )
                stitcher.insertFlowSubimage( flow_lcv, target_meta[ib] ) 
            tcopy = time.time()-tcopy
            timing["+++copy_to_output"] += tcopy
            if verbose:
                print "time to copy results back into full image: ",tcopy," secs"


        # end of while loop
        if verbose:
            print "Processed all the images"

        tout = time.time()
        outputdata.read_entry(ientry)
        stitcher.process( outputdata )
        outputdata.save_entry()
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

    


