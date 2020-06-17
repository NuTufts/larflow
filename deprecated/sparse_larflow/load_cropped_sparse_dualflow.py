import os,sys,time
import ROOT as rt
import numpy as np
from larcv import larcv
from ROOT import std

"""
Load dual flow sparse data
"""
def load_cropped_sparse_dualflow(io,rm_bg_labels=True,
                                 producer="sparsecropdual",
                                 predict_classvec=False,
                                 checkpix=True,
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
    io[larcv.IOManager]    access to IOManager. assumed to have the entry set already.
    rm_bg_labels[bool]     if true, flow values with -4000 is removed
    producer[str]          producer name 
    predict_classvec[bool] if true, provide truth as the target column to flow to
    checkpix[bool]         if true, checks to make sure at least 1 valid flow in each plane, else through out.
    threshold[float]       threshold ADC value for source pixel

    outputs
    -------
    return dictionary with following keys:
      pixadc: numpy array with pixel data in sparse format: 
              (row,col,Y plane ADC, U plane ADC, V plane ADC)
      flowy2u: truth flow Y->U. either pixel shift or target pixel column
      flowy2v: truth flow Y->V. either pixel shift or target pixel column
      masky2u: pixels with good flow has value of 1.0, else 0.0
      masky2v: pixels with good flow has value of 1.0, else 0.0
    """
    
    data    = {}
    verbose = False
    predict_classvec = True

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

    checkpix     = True
    ngoodpix_y2u = 0
    nbadpix_y2u  = 0
    ngoodpix_y2v = 0
    nbadpix_y2v  = 0
    nbadflow_y2u = 0
    nbadflow_y2v = 0
    
    if has_truth:
        data["flowy2u"] = sparse_np[:,5].astype(np.float32).reshape( (sparse_np.shape[0],1) )
        data["flowy2v"] = sparse_np[:,6].astype(np.float32).reshape( (sparse_np.shape[0],1) )

        # produce a good mask
        data["masky2u"] = np.ones( data["flowy2u"].shape, dtype=np.float32 )
        data["masky2v"] = np.ones( data["flowy2v"].shape, dtype=np.float32 )
        data["masky2u"][ data["flowy2u"]<=-4000 ] = 0.
        data["masky2v"][ data["flowy2v"]<=-4000 ] = 0.
        if checkpix:
            nbadpix_y2u  = (data["masky2u"]==0).sum()
            nbadpix_y2v  = (data["masky2v"]==0).sum()
                
        
        # if predicting class vector, we modify the output
        if predict_classvec:
            # add the source columns to the flow
            data["flowy2u"][:,0] += data["pixadc"][:,1]
            data["flowy2v"][:,0] += data["pixadc"][:,1]
            # mask pixels that flow outside target image columns
            data["masky2u"][ data["flowy2u"] < 0 ] = 0
            data["masky2v"][ data["flowy2v"] < 0 ] = 0
            data["masky2u"][ data["flowy2u"] >= 832 ] = 0
            data["masky2v"][ data["flowy2v"] >= 832 ] = 0
            if checkpix:
                nbadflow_y2u = (data["masky2u"]==0).sum()-nbadpix_y2u
                nbadflow_y2v = (data["masky2v"]==0).sum()-nbadpix_y2v

        if checkpix:
            ngoodpix_y2u = data["masky2u"].sum()
            ngoodpix_y2v = data["masky2v"].sum()

    if verbose:
        dtnpmanip += time.time()-tnpmanip
        tottime = time.time()-tottime
        
        print "[load cropped sparse dual flow]"        
        print "  nfeatures=",nfeatures," npts=",sparse_np.shape[0]
        print "  io time: %.3f secs"%(dtio)
        print "  tot time: %.3f secs"%(tottime)
        
        if has_truth and checkpix:
            print "  ngoodpix_y2u=",ngoodpix_y2u," nbadpix_y2u=",nbadpix_y2u," tot=",ngoodpix_y2u+nbadpix_y2u," npts=",sparse_np.shape[0]
            print "  ngoodpix_y2v=",ngoodpix_y2v," nbadpix_y2v=",nbadpix_y2v," tot=",ngoodpix_y2v+nbadpix_y2v," npts=",sparse_np.shape[0]
            if predict_classvec:
                print "  nbadflow_y2u=",nbadflow_y2u,"  nbadflow_y2v=",nbadflow_y2v
            
    if has_truth and checkpix:
        if ngoodpix_y2u==0 or ngoodpix_y2v==0:
            return None

    return data


def load_croppedset_sparse_dualflow_nomc(io,
                                         producer="croppedadc",
                                         threshold=10.0):
    
    # get crop set
    ev_crops = io.get_data( larcv.kProductImage2D, producer )
    crop_v = ev_crops.Image2DArray()
    ncrops = crop_v.size()
    nsets  = ncrops/3

    print "Number of sets=",nsets," ncrops=",ncrops

    thresh_v = std.vector('float')(3,threshold)
    cuton_v  = std.vector('int')(3,1)

    # we are making a batch. collect the sparse arrays
    data = {"pixadc":[]}
    for iset in xrange(nsets):
        # get instance, convert to numpy array, nfeatures per flow
        sparsedata = larcv.SparseImage( crop_v, iset*3, iset*3+2, thresh_v, cuton_v )
        sparse_np  = larcv.as_ndarray( sparsedata, larcv.msg.kNORMAL )
        data["pixadc"].append(sparse_np)
        #print "nfeatures: ",nfeatures
    return data
    
