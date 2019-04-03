from __future__ import print_function
import os,sys
import ROOT as rt
from larcv import larcv
from ublarcvapp import ublarcvapp
from ROOT import std

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
        CheckFlow: true
        MakeCheckImage: false # not working
        SaveTrainingOutput: false
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
    #nentries = 3
    
    out = larcv.IOManager(larcv.IOManager.kWRITE,"")
    out.set_out_file(outputfile)
    out.initialize()

    # when we make the sparse image, we want to produce a number for each pixel
    # where there is charge in the source and target image
    cuton_y2u = std.vector("int")(3,1)
    cuton_y2v = std.vector("int")(3,1)

    thresholds = std.vector("float")(3,10.0)
    thresholds[2] = -3999.0

    for ientry in xrange(nentries):

        print("=======================")
        print("process entry[%d]"%(ientry))
        print("=======================")

        # processor crops larflow
        processor.process_entry(ientry, False, False) # we set autosave to off. else it will clear the data products.
        io = processor.io_mutable()
    
        ev_crops = io.get_data(larcv.kProductImage2D,"detsplit")
        ev_rois  = io.get_data(larcv.kProductROI,"detsplit")

        print("number of random cropped images: %d"%(ev_crops.Image2DArray().size()))
        print("number of random cropped ROIs:   %d"%(ev_rois).ROIArray().size())

        ev_outadc = io.get_data(larcv.kProductImage2D,"croppedadc")
        ev_outflo = io.get_data(larcv.kProductImage2D,"croppedflow")
        print("number of cropped adc images:  %d"%(ev_outadc.Image2DArray().size()))        
        print("number of cropped flow images: %d"%(ev_outflo.Image2DArray().size()))

        # now take images, sparsify and save to disk
        nimgsets = ev_outadc.Image2DArray().size()/3

        # check if number of larflow images makes sense
        if nimgsets!=ev_outflo.Image2DArray().size()/2:
            raise ValueError("number of larflow crops and adc images does not make sense")

        # else make sparse image objects, y2u and y2v
        for iimg in xrange(nimgsets):
            print("Image Crop Set[%d]"%(iimg))
            # make vector of images to sparsify together
            y2u_v = std.vector("larcv::Image2D")()
            y2u_v.push_back( ev_outadc.at(iimg*3+2) )
            y2u_v.push_back( ev_outadc.at(iimg*3+0) )
            y2u_v.push_back( ev_outflo.at(iimg*2+0) )

            y2v_v = std.vector("larcv::Image2D")()
            y2v_v.push_back( ev_outadc.at(iimg*3+2) )
            y2v_v.push_back( ev_outadc.at(iimg*3+1) )
            y2v_v.push_back( ev_outflo.at(iimg*2+1) )

            sparsey2u = larcv.SparseImage( y2u_v, thresholds, cuton_y2u )
            sparsey2v = larcv.SparseImage( y2v_v, thresholds, cuton_y2v )
            npts_y2u = sparsey2u.pixellist().size()/5
            npts_y2v = sparsey2v.pixellist().size()/5
            
            
            print("Sparse Y2U number of points: %d frac=%.2f"%( npts_y2u, float(npts_y2u)/(512.0*832.0)))
            print("Sparse Y2V number of points: %d frac=%.2f"%( npts_y2v, float(npts_y2v)/(512.0*832.0)))

            evout_sparsey2u = out.get_data(larcv.kProductSparseImage, "sparsecropy2u")
            evout_sparsey2v = out.get_data(larcv.kProductSparseImage, "sparsecropy2v")
            evout_sparsey2u.Append( sparsey2u )
            evout_sparsey2v.Append( sparsey2v )
            
            out.set_id( io.event_id().run(), io.event_id().subrun(), io.event_id().event()*100 + iimg )
            out.save_entry()
        io.clear_entry()

    out.finalize()
    processor.finalize()

if __name__ == "__main__":
    """
    run a test example.
    """

    larcv_mctruth     = sys.argv[1]
    output_sparsified = sys.argv[2]
    sparsify_crops( larcv_mctruth, output_sparsified )

    # for testing
    #sparsify_crops( "../../../testdata/mcc9mar_bnbcorsika/larcv_mctruth_ee881c25-aeca-4c92-9622-4c21f492db41.root",
    #                "out_crop_sparsified.root" )
    
