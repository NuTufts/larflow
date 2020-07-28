####################
# INPUT: Image2D
####################
def load_sparse_larflowdata(io,remove_bg_labels=True):
    """
    we need the input data to be a pixel list
    however, the ground truth can be dense arrays

    products returned:
    pixplane[]
    flow[]2[]
    """

    threshold = 10.0
    data = {}

    # profiling variables
    tottime   = time.time()
    dtflow    = 0
    dtconvert = 0
    dtio      = 0
    dtnpmanip = 0

    tio = time.time()
    ev_wire   = io.get_data(larcv.kProductImage2D,"wiremc")
    ev_flow   = io.get_data(larcv.kProductImage2D,"larflow")
    dtio += time.time()-tio

    nimgs = ev_wire.Image2DArray().size()
    meta  = ev_wire.Image2DArray().front().meta()

    if ev_flow.Image2DArray().size()==0:
        has_truth = False
    else:
        has_truth = True

    #flows = [("uflow",0,1,2),("vflow",0,2),("yflow",1,0)]
    flows = [("yflow",2,0,1,(4,5))]


    # cut on ADC values
    for (flowname,src_plane,tar1_plane,tar2_plane,idx) in flows:

        tflowstart = time.time()

        tconvert = time.time()
        srcandtarpix = larcv.as_union_pixelarray( ev_wire.Image2DArray().at(src_plane),
                                                  ev_wire.Image2DArray().at(tar1_plane),
                                                  ev_wire.Image2DArray().at(tar2_plane),
                                                  threshold, larcv.msg.kNORMAL )
        if has_truth:
            flow1 = np.transpose( larcv.as_ndarray( ev_flow.Image2DArray().at(idx[0]) ), (1,0) )
            flow2 = np.transpose( larcv.as_ndarray( ev_flow.Image2DArray().at(idx[1]) ), (1,0) )
        dtconvert += time.time()-tconvert

        #print "src+tar pix: shape=",srcandtarpix.shape
        #print "dense flowimg: shape=",flowimg.shape
        #print "pixellist: ",srcandtarpix[:,0:2].shape
        #print srcandtarpix[:20,0:2]

        tnpmanip  = time.time()
        data["pix"+flowname] = srcandtarpix
        if has_truth:
            data[flowname+"1"]   = flow1[ srcandtarpix[:,0].astype(np.int), srcandtarpix[:,1].astype(int) ]
            data[flowname+"2"]   = flow2[ srcandtarpix[:,0].astype(np.int), srcandtarpix[:,1].astype(int) ]
        else:
            data[flowname+"1"] = None
            data[flowname+"2"] = None
        dtnpmanip += time.time()-tnpmanip

        #data["flow"+flowname]   = larcv.as_pixelarray_with_selection( ev_flow.Image2DArray().at(idx),
        #                                                              ev_wire.Image2DArray().at(src_plane),
        #                                                              threshold, True,
        #                                                              larcv.msg.kNORMAL  )
        # remove no-flow pixels
        #print "flow"+flowname,"=",data["flow"+flowname].shape,
        #print ";srcpix"+flowname,"=",data["srcpix"+flowname].shape,
        #print ";tarpix"+flowname,"=",data["tarpix"+flowname].shape
        dtflow += time.time()-tflowstart

    tottime = time.time()-tottime
    #print "io time: ",dtio
    #print "tot array manip time: ",tottime
    #print "  time for each flow: ",dtflow/len(flows)
    #print "    convert larcv2numpy per flow: ",dtconvert/len(flows)
    #print "    modify numpy arrays: ",(dtnpmanip)/len(flows)


    return data

####################################
# INPUT: SparseImage, single flow
####################################
def load_sparse_larflowdata_sparseimg(io,remove_bg_labels=True):
    """
    we need the input data to be a pixel list
    however, the ground truth can be dense arrays

    products returned:
    pixplane[]
    flow[]2[]
    """

    nflows = 1
    threshold = 10.0
    data = {}

    # profiling variables
    tottime   = time.time()
    dtflow    = 0
    dtconvert = 0
    dtio      = 0
    dtnpmanip = 0

    tio = time.time()
    ev_sparse   = io.get_data(larcv.kProductSparseImage,"larflow_y2u")
    dtio += time.time()-tio

    sparsedata = ev_sparse.at(0)
    sparse_np  = larcv.as_ndarray( sparsedata, larcv.msg.kNORMAL )

    nfeatures = sparsedata.nfeatures()
    meta  = sparsedata.meta_v().front()

    if nflows==2:
        if nfeatures<=3:
            has_truth = False
        else:
            has_truth = True
    else:
        if nfeatures<=2:
            has_truth = False
        else:
            has_truth = True
        

    flows = [ ("yflow",sparse_np, meta) ]

    # cut on ADC values
    for (flowname,sparse_np,meta) in flows:

        tflowstart = time.time()
        #print "src+tar pix: shape=",srcandtarpix.shape
        #print "dense flowimg: shape=",flowimg.shape
        #print "pixellist: ",srcandtarpix[:,0:2].shape
        #print srcandtarpix[:20,0:2]

        tnpmanip  = time.time()
        if nflows==2:
            data["pix"+flowname] = sparse_np[:,0:5] # (row,col,src,tar1,tar2)
            if has_truth:
                data[flowname+"1"]   = sparse_np[:,5].astype(np.float32)
                data[flowname+"2"]   = sparse_np[:,6].astype(np.float32)
            else:
                data[flowname+"1"] = None
                data[flowname+"2"] = None
            dtnpmanip += time.time()-tnpmanip
        elif nflows==1:
            data["pix"+flowname] = sparse_np[:,0:4] # (row,col,src,tar)
            if has_truth:
                data[flowname+"1"] = sparse_np[:,4].astype(np.float32) # (truth1)
                data[flowname+"2"] = None                
            else:
                data[flowname+"1"] = None
                data[flowname+"2"] = None                
            dtnpmanip += time.time()-tnpmanip
            

        #data["flow"+flowname]   = larcv.as_pixelarray_with_selection( ev_flow.Image2DArray().at(idx),
        #                                                              ev_wire.Image2DArray().at(src_plane),
        #                                                              threshold, True,
        #                                                              larcv.msg.kNORMAL  )
        # remove no-flow pixels
        #print "flow"+flowname,"=",data["flow"+flowname].shape,
        #print ";srcpix"+flowname,"=",data["srcpix"+flowname].shape,
        #print ";tarpix"+flowname,"=",data["tarpix"+flowname].shape
        dtflow += time.time()-tflowstart

    tottime = time.time()-tottime
    #print "io time: ",dtio
    #print "tot array manip time: ",tottime
    #print "  time for each flow: ",dtflow/len(flows)
    #print "    convert larcv2numpy per flow: ",dtconvert/len(flows)
    #print "    modify numpy arrays: ",(dtnpmanip)/len(flows)


    return data

def load_larflow_larcvdata( name, inputfile, batchsize, nworkers,
                            nflows=2,
                            tickbackward=False, readonly_products=None,
                            producer_name="larflow"):
    feeder = SparseLArFlowPyTorchDataset(inputfile, batchsize,
                                         tickbackward=tickbackward, nworkers=nworkers, nflows=nflows,
                                         producer_name=producer_name,
                                         readonly_products=readonly_products,
                                         feedername=name)
    return feeder


###########################################
# INPUT: Cropped SparseImage, separated
###########################################
def load_sparsecropped_larflowdata_sparseimg(io,rm_bg_labels=True,
                                             dualflow=True, producer="sparsecrop",
                                             flowdir="y2u",threshold=10.0):
    """
    we need the input data to be a pixel list for Sparse Convolutions
    
    we import cropped sparseimage objects from a LArCV rootfile

    the sparseimg is assumed to have at most 3 features
     0: source adc
     1: target adc
     2: mc only. truth flow, -4000 for invalid flow
    
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

    threshold = 10.0
    data = {}

    # profiling variables
    tottime   = time.time()
    dtflow    = 0
    dtconvert = 0
    dtio      = 0
    dtnpmanip = 0

    # get data from iomanager
    tio = time.time()
    ev_sparse = {}
    flows = ['y2u','y2v']
    if not dualflow:
        flows = [flowdir]
    for fl in flows:
        producername = "%s%s"%(producer,fl)
        ev_sparse[fl] = io.get_data(larcv.kProductSparseImage,producername)
    dtio += time.time()-tio

    # get instance, convert to numpy array, nfeatures per flow
    # numpy array is (N,5) with 2nd dimension is (row,col,source_adc,target_adc,truth_flow)
    sparsedata = {}
    sparse_np  = {}
    nfeatures  = {}
    for fl in flows:
        sparsedata[fl] = ev_sparse[fl].at(0)
        sparse_np[fl]  = larcv.as_ndarray( sparsedata[fl], larcv.msg.kNORMAL )
        nfeatures[fl]  = sparsedata[fl].nfeatures()

    # source meta, same for flow directions
    meta  = sparsedata[ flows[0] ].meta_v().front()

    # has truth, 3rd feature is truth flow
    has_truth = False
    if nfeatures[ flows[0] ]==3:
        has_truth = True        

    flowdata = []
    for fl in flows:
        flowdata.append( (fl, sparse_np[fl], meta) )

    # cut on ADC values
    for (flowname,sparsemat,meta) in flowdata:

        tflowstart = time.time()
        print "src+tar pix: shape=",sparsemat.shape
        #print "dense flowimg: shape=",flowimg.shape
        #print "pixellist: ",srcandtarpix[:,0:2].shape
        #print srcandtarpix[:20,0:2]

        tnpmanip  = time.time()
        data["pix"+flowname] = sparsemat[:,0:4] # (row,col,src,tar,flow)
        if has_truth:
            data["flow"+flowname]   = sparsemat[:,4].astype(np.float32).reshape( (sparsemat.shape[0],1) )
        dtnpmanip += time.time()-tnpmanip
            

        #data["flow"+flowname]   = larcv.as_pixelarray_with_selection( ev_flow.Image2DArray().at(idx),
        #                                                              ev_wire.Image2DArray().at(src_plane),
        #                                                              threshold, True,
        #                                                              larcv.msg.kNORMAL  )
        # remove no-flow pixels
        #print "flow"+flowname,"=",data["flow"+flowname].shape,
        #print ";srcpix"+flowname,"=",data["srcpix"+flowname].shape,
        #print ";tarpix"+flowname,"=",data["tarpix"+flowname].shape
        dtflow += time.time()-tflowstart

    tottime = time.time()-tottime
    #print "io time: ",dtio
    #print "tot array manip time: ",tottime
    #print "  time for each flow: ",dtflow/len(flows)
    #print "    convert larcv2numpy per flow: ",dtconvert/len(flows)
    #print "    modify numpy arrays: ",(dtnpmanip)/len(flows)


    return data

def load_larflow_larcvdata( name, inputfile, batchsize, nworkers,
                            nflows=2,
                            tickbackward=False, readonly_products=None,
                            producer_name="larflow"):
    feeder = SparseLArFlowPyTorchDataset(inputfile, batchsize,
                                         tickbackward=tickbackward, nworkers=nworkers, nflows=nflows,
                                         producer_name=producer_name,
                                         readonly_products=readonly_products,
                                         feedername=name)
    return feeder
