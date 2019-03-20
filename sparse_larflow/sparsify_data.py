import os,sys
import ROOT as rt
from larcv import larcv
from ROOT import std



def sparsify(inputfile, outputfile,
             adc_producer="wiremc", flow_producer="larflow"):
    io = larcv.IOManager(larcv.IOManager.kREAD,"",larcv.IOManager.kTickBackward)
    io.add_in_file(inputfile)
    io.specify_data_read(larcv.kProductImage2D,adc_producer)
    io.specify_data_read(larcv.kProductImage2D,flow_producer)
    io.initialize()

    out = larcv.IOManager(larcv.IOManager.kWRITE,"")
    out.set_out_file(outputfile)
    out.initialize()

    flowdef_list = [(2,0,1,4,5)] # (src,tar1,tar2,flow-index-1,flow-index-2)

    for ientry in xrange(io.get_n_entries()):
        io.read_entry(ientry)

        ev_adc  = io.get_data(larcv.kProductImage2D,"wiremc")
        ev_flow = io.get_data(larcv.kProductImage2D,"larflow")
        adc_v  = ev_adc.Image2DArray()
        flow_v = ev_flow.Image2DArray()

        # for larflow, we pack 5 images together
        # 1) src image
        # 2) target 1 image
        # 3) target 2 image
        # 4) src->target1 flow
        # 5) src->target2 flow
        threshold_v = std.vector("float")(5,5.0)
        cuton_pixel_v = std.vector("int")(5,0)
        cuton_pixel_v[0] = 1
        cuton_pixel_v[1] = 1
        cuton_pixel_v[2] = 1
        flowset_v = std.vector("larcv::Image2D")()
        for (srcidx,tar1idx,tar2idx,flow1idx,flow2idx) in flowdef_list:
            flowset_v.push_back( adc_v.at(srcidx) )
            flowset_v.push_back( adc_v.at(tar1idx) )
            flowset_v.push_back( adc_v.at(tar2idx) )
            flowset_v.push_back( flow_v.at(flow1idx) )
            flowset_v.push_back( flow_v.at(flow2idx) )

        adc_sparse_tensor = larcv.SparseImage(flowset_v,threshold_v)
        print "number of sparse floats: ",adc_sparse_tensor.pixellist().size()

        ev_sparse  = out.get_data(larcv.kProductSparseImage,"larflow")

        sparse_nd = larcv.as_ndarray(adc_sparse_tensor,larcv.msg.kDEBUG)


        ncols = adc_v.front().meta().cols()
        nrows = adc_v.front().meta().rows()
        maxpixels = ncols*nrows
        occupancy_frac = float(sparse_nd.shape[0])/maxpixels

        print "SparseImage shape: ",sparse_nd.shape," occupancy=",occupancy_frac


        ev_sparse.Append( adc_sparse_tensor )

        out.set_id( io.event_id().run(),
                    io.event_id().subrun(),
                    io.event_id().event() )
        out.save_entry()
        print "Filled Event %d"%(ientry)
        #break

    out.finalize()
    io.finalize()

if __name__ == "__main__":
    """
    run a test example.
    """

    sparsify( "../testdata/mcc9mar_bnbcorsika/larcv_mctruth_ee881c25-aeca-4c92-9622-4c21f492db41.root",
              "out_sparsified.root" )
