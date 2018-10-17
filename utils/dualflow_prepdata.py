import os,sys

## =========================================================
## FUNCTIONS FOR LOADING DUAL FLOW DATA
## =========================================================
def load_dualflow_data_wtruth_larcv2( io ):
    from larcv import larcv
    import numpy as np
    
    width  = 832
    height = 512
    src_adc_threshold = 10.0

    ev_adc = io.get_data("image2d","adc")
    ev_flo = io.get_data("image2d","pixflow")
    ev_vis = io.get_data("image2d","pixvisi")
    
    index = (1,0)
    products = ["source","targetu","targetv","flowy2u","flowy2v","visiy2u","visiy2v","meta"]
    data = {}
    for k in products:
        if k !="meta":
            data[k] = np.zeros( (1,width,height), dtype=np.float32 )
        else:
            data[k] = np.zeros( (3,width,height), dtype=np.float32 )            
        
    data["source"][0,:,:]  = larcv.as_ndarray( ev_adc.as_vector()[2] ).transpose(1,0)
    data["targetu"][0,:,:] = larcv.as_ndarray( ev_adc.as_vector()[0] ).transpose(1,0)
    data["targetv"][0,:,:] = larcv.as_ndarray( ev_adc.as_vector()[1] ).transpose(1,0)

    data["flowy2u"][0,:,:] = larcv.as_ndarray( ev_flo.as_vector()[0] ).transpose(1,0)
    data["flowy2v"][0,:,:] = larcv.as_ndarray( ev_flo.as_vector()[1] ).transpose(1,0)

    data["visiy2u"][0,:,:] = larcv.as_ndarray( ev_vis.as_vector()[0] ).transpose(1,0)
    data["visiy2v"][0,:,:] = larcv.as_ndarray( ev_vis.as_vector()[1] ).transpose(1,0)

    for ip in xrange(0,3):
        data["meta"][ip,0,0] = ev_adc.as_vector()[ip].meta().min_x()
        data["meta"][ip,0,1] = ev_adc.as_vector()[ip].meta().min_y()
        data["meta"][ip,0,2] = ev_adc.as_vector()[ip].meta().max_x()
        data["meta"][ip,0,3] = ev_adc.as_vector()[ip].meta().max_y()
            
    return data

def prep_data_pytorch( ioserver, batchsize, width, height, src_adc_threshold, device ):
    """
    Turns data loaded from above functions into pytorch tensors
    """

    #print "PREP DATA: ",train_or_valid,"GPUMODE=",GPUMODE,"GPUID=",GPUID    

    # get data
    data = ioserver.get_batch_dict()

    # make torch tensors from numpy arrays
    index = (0,1,3,2)
    #print "prep_data: ",data.keys()
    #print "source shape: ",data["source_%s"%(train_or_valid)].shape
    source_t  = torch.from_numpy( data["source"] ).to( device=device )   # source image ADC
    target1_t = torch.from_numpy( data["targetu"] ).to(device=device )   # target image ADC
    target2_t = torch.from_numpy( data["targetv"] ).to( device=device )  # target2 image ADC
    flow1_t   = torch.from_numpy( data["flowy2u"] ).to( device=device )   # flow from source to target
    flow2_t   = torch.from_numpy( data["flowy2v"] ).to( device=device ) # flow from source to target
    fvisi1_t  = torch.from_numpy( data["visiy2u"] ).to( device=device )  # visibility at source (float)
    fvisi2_t  = torch.from_numpy( data["visiy2v"] ).to( device=device ) # visibility at source (float)

    # apply threshold to source ADC values. returns a byte mask
    fvisi1_t  = fvisi1_t.clamp(0.0,1.0)
    fvisi2_t  = fvisi2_t.clamp(0.0,1.0)

    # make integer visi
    visi1_t = fvisi1_t.reshape( (batchsize,fvisi1_t.size()[2],fvisi1_t.size()[3]) ).long()
    visi2_t = fvisi2_t.reshape( (batchsize,fvisi2_t.size()[2],fvisi2_t.size()[3]) ).long()

    # image column origins
    meta_data_np = data["meta"]
    #print meta_data_np
    source_minx  = torch.from_numpy( meta_data_np[:,2,0,0].reshape((batchsize)) ).to(device=device)
    target1_minx = torch.from_numpy( meta_data_np[:,0,0,0].reshape((batchsize)) ).to(device=device)
    target2_minx = torch.from_numpy( meta_data_np[:,1,0,0].reshape((batchsize)) ).to(device=device)
    
    return source_t, target1_t, target2_t, flow1_t, flow2_t, visi1_t, visi2_t, fvisi1_t, fvisi2_t, source_minx, target1_minx, target2_minx
