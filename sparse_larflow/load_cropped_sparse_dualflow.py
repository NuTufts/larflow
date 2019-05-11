import os,sys,time
import ROOT as rt
import numpy as np
from larcv import larcv

"""
Load dual flow sparse data
"""
def load_cropped_sparse_dualflow(io,rm_bg_labels=True,
                                 producer="sparsecropdual",
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
    
    data = {}

    # profiling variables
    tottime   = time.time()
    dtflow    = 0
    dtconvert = 0
    dtio      = 0
    dtnpmanip = 0

    # get data from iomanager
    tio = time.time()
    ev_sparse = io.get_data(larcv.kProductSparseImage,producer)
    dtio += time.time()-tio

    # get instance, convert to numpy array, nfeatures per flow
    # numpy array is (N,7) with 2nd dimension is (row,col,source_adc,target_adc,truth_flow)
    sparsedata = ev_sparse.at(0)
    sparse_np  = larcv.as_ndarray( sparsedata, larcv.msg.kNORMAL )
    nfeatures  = sparsedata.nfeatures()
    #print "nfeatures: ",nfeatures
    
    # source meta, same for flow directions
    meta  = sparsedata.meta_v().front()

    # has truth, 3rd feature is truth flow
    has_truth = False
    if nfeatures==5:
        has_truth = True        


    tnpmanip  = time.time()
    data["pixadc"] = sparse_np[:,0:5] # (row,col,src,tar,tar1)
    if has_truth:
        data["flowy2u"] = sparse_np[:,5].astype(np.float32).reshape( (sparse_np.shape[0],1) )
        data["flowy2v"] = sparse_np[:,6].astype(np.float32).reshape( (sparse_np.shape[0],1) )        
    dtnpmanip += time.time()-tnpmanip
    tottime = time.time()-tottime
    #print "io time: ",dtio
    #print "tot array manip time: ",tottime
    #print "  time for each flow: ",dtflow/len(flows)
    #print "    convert larcv2numpy per flow: ",dtconvert/len(flows)
    #print "    modify numpy arrays: ",(dtnpmanip)/len(flows)


    return data


