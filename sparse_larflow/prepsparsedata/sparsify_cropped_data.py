from __future__ import print_function
import os,sys
import ROOT as rt
from larcv import larcv
from ublarcvapp import ublarcvapp
from ROOT import std

def gen_default_split_cfg(wire_producer):
    split_cfg="""Verbosity: 2
    InputProducer: "%s"
    OutputBBox2DProducer: "detsplit"
    CropInModule: true
    OutputCroppedProducer: "detsplit"
    BBoxPixelHeight: 512
    BBoxPixelWidth: 832
    CoveredZWidth: 310
    FillCroppedYImageCompletely: true
    DebugImage: false
    MaxImages: -1
    RandomizeCrops: false
    MaxRandomAttempts: 50
    MinFracPixelsInCrop: 0.01
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

def sparsify_crops(inputfile, outputfile,
                   adc_producer="wiremc", flow_producer="larflow",
                   split_config=None,
                   flowdirs=['y2u','y2v']):

    nflows = len(flowdirs)
    
    io = larcv.IOManager(larcv.IOManager.kREAD,"",larcv.IOManager.kTickBackward)
    io.add_in_file(inputfile)
    io.specify_data_read(larcv.kProductImage2D,adc_producer)
    io.specify_data_read(larcv.kProductImage2D,flow_producer)
    io.initialize()

    split_cfg  = gen_default_split_cfg(adc_producer)
    lfcrop_cfg = gen_default_larflowcrop_cfg()

    split_algo = ublarcvapp.UBSplitDetector()
    print(split_cfg,file=open("ubsplit.cfg",'w'))
    split_pset = larcv.CreatePSetFromFile( "ubsplit.cfg", "UBSplitDetector" )    
    split_algo.configure(split_pset)
    split_algo.initialize()
    
    out = larcv.IOManager(larcv.IOManager.kWRITE,"")
    out.set_out_file(outputfile)
    out.initialize()

    for ientry in xrange(io.get_n_entries()):
        io.read_entry(ientry)

        ev_adc  = io.get_data(larcv.kProductImage2D,"wiremc")
        ev_flow = io.get_data(larcv.kProductImage2D,"larflow")
        adc_v  = ev_adc.Image2DArray()
        flow_v = ev_flow.Image2DArray()

        # random splits
        crops_v = std.vector("larcv::Image2D")()
        rois_v  = std.vector("larcv::ROI")()
        split_algo.process(adc_v, crops_v, rois_v)
        print("number of random crops: %d"%(rois_v.size()))

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
    io.finalize()

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
    
