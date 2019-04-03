from __future__ import print_function
import os,sys
import ROOT as rt
from larcv import larcv
from ublarcvapp import ublarcvapp
from ROOT import std

def gen_default_split_cfg(wire_producer):
    split_cfg="""Verbosity: 0
    InputProducer: "%s"
    OutputBBox2DProducer: "detsplit"
    CropInModule: true
    OutputCroppedProducer: "detsplit"
    BBoxPixelHeight: 512
    BBoxPixelWidth: 832
    CoveredZWidth: 310
    FillCroppedYImageCompletely: true
    DebugImage: false
    MaxImages: 3
    RandomizeCrops: true
    MaxRandomAttempts: 50
    MinFracPixelsInCrop: 0.0001
    """%(wire_producer)
    return split_cfg

def gen_default_larflowcrop_cfg():
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
    CheckFlow:  true
    MakeCheckImage: true
    DoMaxPool: false
    RowDownsampleFactor: 2
    ColDownsampleFactor: 2
    MaxImages: -1
    LimitOverlap: false
    RequireMinGoodPixels: false
    MaxOverlapFraction: 0.2
    IsMC: true
    UseVectorizedCode: true
    """
    return lfcrop_cfg

def default_processor_config(input_adc_producer,input_chstatus_producer):
    processor_cfg="""ProcessDriver: {
    Verbosity: 0
    EnableFilter: false
    RandomAccess: false
    ProcessType: ["UBSplitDetector","UBCropLArFlow"]
    ProcessName: ["Split","Crop"]

    IOManager: {
      Verbosity: 2
      Name: "IOManager"
      IOMode: 2
      OutFileName: "test_crop_larflow.root"
      InputFiles: []
      InputDirs: []
      StoreOnlyType: []
      StoreOnlyName: []
      TickBackward: true
    }

    ProcessList: {
      Split: {
        Verbosity: 0
        InputProducer: "%s"
        OutputBBox2DProducer: "detsplit"
        CropInModule: true
        OutputCroppedProducer: "detsplit"
        BBoxPixelHeight: 512
        BBoxPixelWidth: 832
        CoveredZWidth: 310
        FillCroppedYImageCompletely: true
        DebugImage: false
        MaxImages: 20
        RandomizeCrops: true
        MaxRandomAttempts: 50
        MinFracPixelsInCrop: 0.0001
      }
      Crop: {
        Verbosity:0
        InputADCProducer: "%s"
        InputBBoxProducer: "detsplit"
        InputCroppedADCProducer: "detsplit"
        InputVisiProducer: ""
        InputFlowProducer: "%s"
        InputChStatusProducer: "%s"
        OutputCroppedADCProducer: "croppedadc"
        OutputCroppedVisiProducer: "croppedvisi"
        OutputCroppedFlowProducer: "croppedflow"
        OutputCroppedMetaProducer: "croppedmeta_defunct"
        IsMC: true
        HasVisibilityImage: false
        UseVectorizedCode: true # always true now
        MaxImages: 10
        Thresholds: [10.0,10.0,10.0]
        DoMaxPool: false # not working
        RowDownsampleFactor: 1 # not working
        ColDownsampleFactor: 1 # not working
        RequireMinGoodPixels: true # not working
        LimitOverlap: true
        MaxOverlapFraction: 0.25
        CheckFlow: false # not working
        MakeCheckImage: false # not working
        SaveTrainingOutput: true
        OutputFilename: "croppedout.root"
      }
    }    
}"""%(input_adc_producer,input_adc_producer,"larflow",input_chstatus_producer)
    return processor_cfg
    

def sparsify_crops(inputfile, outputfile,
                   adc_producer="wiremc", flow_producer="larflow",
                   chstatus_producer="wiremc",
                   split_config=None,
                   flowdirs=['y2u','y2v']):

    nflows = len(flowdirs)

    # create a processor

    # first create cfg file
    processor_cfg = default_processor_config(adc_producer,"wiremc")
    print(processor_cfg,file=open("cropflow_processor.cfg",'w'))
    processor = larcv.ProcessDriver( "ProcessDriver" )
    processor.configure( "cropflow_processor.cfg" )
    io = processor.io()
    io.add_in_file(inputfile)
    processor.initialize()

    nentries = processor.io().get_n_entries()
    
    out = larcv.IOManager(larcv.IOManager.kWRITE,"")
    out.set_out_file(outputfile)
    out.initialize()

    for ientry in xrange(nentries):

        print("=======================")
        print("process entry[%d]"%(ientry))
        print("=======================")
        
        processor.process_entry(ientry, False, False) # we set autosave to off. else it will clear the data products.
        #processor.process_entry(ientry)
        io = processor.io_mutable()
        

        #ev_adc  = io.get_data(larcv.kProductImage2D,"wiremc")
        #ev_flow = io.get_data(larcv.kProductImage2D,"larflow")
        #adc_v  = ev_adc.Image2DArray()
        #flow_v = ev_flow.Image2DArray()


        ev_crops = io.get_data(larcv.kProductImage2D,"detsplit")
        ev_rois  = io.get_data(larcv.kProductROI,"detsplit")
        # random splits
        #crops_v = std.vector("larcv::Image2D")()
        #rois_v  = std.vector("larcv::ROI")()
        #split_algo.process(adc_v, crops_v, rois_v)
        print("number of random cropped images: %d"%(ev_crops.Image2DArray().size()))
        print("number of random cropped ROIs:   %d"%(ev_rois).ROIArray().size())

        ev_outflo = io.get_data(larcv.kProductImage2D,"croppedflow")
        print("number of cropped flow images: %d"%(ev_outflo.Image2DArray().size()))
        
        # processok = split_algo.process( adc_v, crops_v, rois_v )

        # # for larflow, we pack up to 5 images together, depending on flows
        # # if flowdirs=['y2u','y2v']:
        # #   1) src image
        # #   2) target 1 image
        # #   3) target 2 image
        # #   4) src->target1 flow
        # #   5) src->target2 flow
        # # if flowdirs=['y2u']:
        # #   1) src image
        # #   2) target 1 image
        # #   3) src->target1 flow
        # # if flowdirs=['y2v']:
        # #   1) src image
        # #   2) target 2 image
        # #   3) src->target2 flow
        

        # if nflows==2:
        #     nimgs = 5
        # else:
        #     nimgs = 3
            
        # threshold_v = std.vector("float")(nimgs,5.0)
        # cuton_pixel_v = std.vector("int")(nimgs,0)        
        # cuton_pixel_v[0] = 1
        # cuton_pixel_v[1] = 1
        # if nflows==2:
        #     cuton_pixel_v[2] = 1
            
        # flowset_v = std.vector("larcv::Image2D")()
        # for (srcidx,tar1idx,tar2idx,flow1idx,flow2idx) in flowdef_list:
        #     if nflows==2:
        #         flowset_v.push_back( adc_v.at(srcidx) )
        #         flowset_v.push_back( adc_v.at(tar1idx) )
        #         flowset_v.push_back( adc_v.at(tar2idx) )
        #         flowset_v.push_back( flow_v.at(flow1idx) )
        #         flowset_v.push_back( flow_v.at(flow2idx) )
        #     elif nflows==1 and flowdirs[0]=='y2u':
        #         flowset_v.push_back( adc_v.at(srcidx) )
        #         flowset_v.push_back( adc_v.at(tar1idx) )
        #         flowset_v.push_back( flow_v.at(flow1idx) )
        #     elif nflows==1 and flowdirs[0]=='y2v':
        #         flowset_v.push_back( adc_v.at(srcidx) )
        #         flowset_v.push_back( adc_v.at(tar2idx) )
        #         flowset_v.push_back( flow_v.at(flow2idx) )
                

        # adc_sparse_tensor = larcv.SparseImage(flowset_v,threshold_v,cuton_pixel_v)
        # print "number of sparse floats: ",adc_sparse_tensor.pixellist().size()

        # producername = "larflow"
        # if nflows==1:
        #     producername += "_"+flowdirs[0]
        #     ev_sparse  = out.get_data(larcv.kProductSparseImage,producername)

        # sparse_nd = larcv.as_ndarray(adc_sparse_tensor,larcv.msg.kDEBUG)


        # ncols = adc_v.front().meta().cols()
        # nrows = adc_v.front().meta().rows()
        # maxpixels = ncols*nrows
        # occupancy_frac = float(sparse_nd.shape[0])/maxpixels

        # print "SparseImage shape: ",sparse_nd.shape," occupancy=",occupancy_frac


        # ev_sparse.Append( adc_sparse_tensor )

        # out.set_id( io.event_id().run(),
        #             io.event_id().subrun(),
        #             io.event_id().event() )
        # out.save_entry()
        # print "Filled Event %d"%(ientry)
        break

    out.finalize()
    processor.finalize()

if __name__ == "__main__":
    """
    run a test example.
    """

    #larcv_mctruth     = sys.argv[1]
    #output_sparsified = sys.argv[2]
    
    sparsify_crops( "../../../testdata/mcc9mar_bnbcorsika/larcv_mctruth_ee881c25-aeca-4c92-9622-4c21f492db41.root",
                    "out_crop_sparsified.root" )

    #sparsify( larcv_mctruth, output_sparsified, flowdirs=['y2u','y2v'] )

    #output_sparsified_y2u = output_sparsified.replace(".root","_y2u.root")
    #sparsify( larcv_mctruth, output_sparsified_y2u, flowdirs=['y2u'] )

    #output_sparsified_y2v = output_sparsified.replace(".root","_y2v.root")
    #sparsify( larcv_mctruth, output_sparsified_y2v, flowdirs=['y2v'] )
    
