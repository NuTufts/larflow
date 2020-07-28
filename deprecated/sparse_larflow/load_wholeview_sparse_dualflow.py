import os,sys,time
import ROOT as rt
import numpy as np
from larcv import larcv
from ROOT import std

"""
Load dual flow sparse data
"""
def load_wholeview_sparse_dualflow(io,rm_bg_labels=True,
                                   adc_producer="wiremc",
                                   larflow_producer="larflow",
                                   flowdirs=['y2u','y2v'],
                                   threshold=10.0):
    """
    we need the input data to be a pixel list for Sparse Convolutions
    
    we import cropped sparseimage objects from a LArCV rootfile

    the sparseimg is assumed to have at 7 features
    (0,1): (row,col)
    2: source  adc, Y plane
    3: target1 adc, U plane
    4: target2 adc, V plane
    5: flow, Y->U (mc only)
    6: flow, Y->V (mc only)

    For MC, -4000 is for invalid flow
    
    inputs
    ------
    io[larcv.IOManager] access to IOManager. assumed to have the entry set already.
    rm_bg_labels[bool]  if true, flow values with -4000 is removed
    dualflow[bool]      if true, loads both 'y2u' or 'y2v' flows; if false, load flowdir direction
    producer[str]       producer name 
    flowdir[str]        'y2u' or 'y2v'; relevant if dualflow is False
    threshold[float]    threshold ADC value for source pixel

    outputs
    -------

    """
    
    data    = {}
    verbose = False

    # profiling variables
    tottime   = time.time()
    dtflow    = 0
    dtconvert = 0
    dtio      = 0
    dtnpmanip = 0

    # get data from iomanager
    tio = time.time()
    ev_img = io.get_data(larcv.kProductImage2D,    adc_producer)
    adc_v = ev_img.Image2DArray()    
    if larflow_producer is not None:
        ev_flow   = io.get_data(larcv.kProductImage2D, larflow_producer)
        flow_v    = ev_flow.Image2DArray()
    else:
        flow_v = None

    dtio += time.time()-tio

    # sparsify image
    nimgs = adc_v.size()
    if flow_v is not None:
        nimgs += 2
        
    meta  = flow_v.at(2).meta()


    nflows = len(flowdirs)
    flowdef_list = [(2,0,1,4,5)] # (src,tar1,tar2,flow-index-1,flow-index-2)
    flows = [('dual',2,0,1,(4,5))]

    for (flowname,src_plane,tar1_plane,tar2_plane,idx) in flows:

        tconvert = time.time()

        threshold_v = std.vector("float")(nimgs,threshold)
        cuton_pixel_v = std.vector("int")(nimgs,0)        
        cuton_pixel_v[0] = 1
        cuton_pixel_v[1] = 1
        cuton_pixel_v[2] = 1
            
        flowset_v = std.vector("larcv::Image2D")()
        for (srcidx,tar1idx,tar2idx,flow1idx,flow2idx) in flowdef_list:
            if nflows==2:
                flowset_v.push_back( adc_v.at(srcidx) )
                flowset_v.push_back( adc_v.at(tar1idx) )
                flowset_v.push_back( adc_v.at(tar2idx) )
                flowset_v.push_back( flow_v.at(flow1idx) )
                flowset_v.push_back( flow_v.at(flow2idx) )
            elif nflows==1 and flowdirs[0]=='y2u':
                flowset_v.push_back( adc_v.at(srcidx) )
                flowset_v.push_back( adc_v.at(tar1idx) )
                flowset_v.push_back( flow_v.at(flow1idx) )
            elif nflows==1 and flowdirs[0]=='y2v':
                flowset_v.push_back( adc_v.at(srcidx) )
                flowset_v.push_back( adc_v.at(tar2idx) )
                flowset_v.push_back( flow_v.at(flow2idx) )
                

        adc_sparse_tensor = larcv.SparseImage(flowset_v,threshold_v,cuton_pixel_v)
        print "number of sparse floats: ",adc_sparse_tensor.pixellist().size()

        sparse_np = larcv.as_ndarray(adc_sparse_tensor,larcv.msg.kDEBUG)
        data["pixadc"]  = sparse_np[:,0:5]
        data["flowy2u"] = sparse_np[:,5].astype(np.float32).reshape( (sparse_np.shape[0],1) )
        data["flowy2v"] = sparse_np[:,6].astype(np.float32).reshape( (sparse_np.shape[0],1) )
        
        #dtnpmanip += time.time()-tnpmanip

        #data["flow"+flowname]   = larcv.as_pixelarray_with_selection( ev_flow.Image2DArray().at(idx),
        #                                                              ev_wire.Image2DArray().at(src_plane),
        #                                                              threshold, True,
        #                                                              larcv.msg.kNORMAL  )
        # remove no-flow pixels
        #print "flow"+flowname,"=",data["flow"+flowname].shape,
        #print ";srcpix"+flowname,"=",data["srcpix"+flowname].shape,
        #print ";tarpix"+flowname,"=",data["tarpix"+flowname].shape
        dtconvert += time.time()-tconvert

    tottime = time.time()-tottime
    #print "io time: ",dtio
    #print "tot array manip time: ",tottime
    #print "  time for each flow: ",dtflow/len(flows)
    #print "    convert larcv2numpy per flow: ",dtconvert/len(flows)
    #print "    modify numpy arrays: ",(dtnpmanip)/len(flows)
    
    return data


