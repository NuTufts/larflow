import os,sys
from larcv import larcv
import torch
import ROOT as rt
import cv2 as cv
from serverfeed.larcv2socketfeed import LArCV2SocketFeeder
from serverfeed.larcv2server import LArCV2Server
import numpy as np

rt.gStyle.SetOptStat(0)
rt.gStyle.SetPadRightMargin(0.2)

"""
Script to study consistency loss behavior.

we load a subimage.
push it through the model.
record the 3d consistency for each pixel that 
  contributes to the loss.
visualize this info
"""

sys.path.append("../models/")
from larflow_uresnet_mod3 import LArFlowUResNet
from larflow_consistency3d_loss import LArFlow3DConsistencyLoss

## =========================================================
## MODEL/LOSS
## =========================================================

# create model, mark it to run on the GPU
CHECKPOINT_FILE="checkpoint.2500th.tar"
map_location={"cuda:0":"cpu","cuda:1":"cpu"}
DEVICE_IDS=[0,0]
DEVICE=torch.device("cuda:0")
model = LArFlowUResNet( num_classes=2, input_channels=1,
                        layer_channels=[16,32,64,128],
                        layer_strides= [ 2, 2, 2,  2],
                        num_final_features=64,
                        use_deconvtranspose=False,
                        onlyone_res=True,
                        showsizes=False,
                        use_visi=False,
                        use_grad_checkpoints=True,
                        gpuid1=DEVICE_IDS[0],
                        gpuid2=DEVICE_IDS[0] )

torch.load( CHECKPOINT_FILE, map_location=map_location )
model.to(device=DEVICE)
model.eval()

loss = LArFlow3DConsistencyLoss(832,512,1, return_pos_images=True, reduce=False)

inputfile = "../testdata/smallsample/larcv_dlcosmictag_5482426_95_smallsample082918.root"

## =========================================================
## DATA PREP
## =========================================================
batchsize_valid=1
IMAGE_WIDTH=832
IMAGE_HEIGHT=512
ADC_THRESH=10
VALID_LARCV_CONFIG="test_flowloader_832x512_dualflow_valid.cfg"
#iovalid = LArCV2SocketFeeder(batchsize_valid,"valid",VALID_LARCV_CONFIG,"ThreadProcessorValid",1,port=1)
#iovalid = LArCV2Server(batchsize_valid,"valid",VALID_LARCV_CONFIG,"ThreadProcessorValid",1,port=1)
def prep_data( larcvloader, train_or_valid, batchsize, width, height, src_adc_threshold, device ):
    """
    inputs
    ------
    larcvloader: instance of LArCVDataloader
    train_or_valid (str): "train" or "valid"
    batchsize (int)
    width (int)
    height(int)
    src_adc_threshold (float)

    outputs
    -------
    source_var (Pytorch Variable): source ADC
    targer_var (Pytorch Variable): target ADC
    flow_var (Pytorch Variable): flow from source to target
    visi_var (Pytorch Variable): visibility of source (long)
    fvisi_var(Pytorch Variable): visibility of target (float)
    """

    #print "PREP DATA: ",train_or_valid,"GPUMODE=",GPUMODE,"GPUID=",GPUID    

    # get data
    #data = larcvloader[0]
    data = larcvloader.get_batch_dict()

    # make torch tensors from numpy arrays
    index = (0,1,3,2)
    #print "prep_data: ",data.keys()
    #print "source shape: ",data["source_%s"%(train_or_valid)].shape
    source_t  = torch.from_numpy( data["source_%s"%(train_or_valid)].reshape(batchsize,1,height,width).transpose(index) ).to( device=device )   # source image ADC
    target1_t = torch.from_numpy( data["target1_%s"%(train_or_valid)].reshape(batchsize,1,height,width).transpose(index) ) .to(device=device )   # target image ADC
    target2_t = torch.from_numpy( data["target2_%s"%(train_or_valid)].reshape(batchsize,1,height,width).transpose(index) ) .to( device=device )  # target2 image ADC
    flow1_t   = torch.from_numpy( data["pixflow1_%s"%(train_or_valid)].reshape(batchsize,1,height,width).transpose(index) ).to( device=device )   # flow from source to target
    flow2_t   = torch.from_numpy( data["pixflow2_%s"%(train_or_valid)].reshape(batchsize,1,height,width).transpose(index) ).to( device=device ) # flow from source to target
    fvisi1_t  = torch.from_numpy( data["pixvisi1_%s"%(train_or_valid)].reshape(batchsize,1,height,width).transpose(index) ).to( device=device )  # visibility at source (float)
    fvisi2_t  = torch.from_numpy( data["pixvisi2_%s"%(train_or_valid)].reshape(batchsize,1,height,width).transpose(index) ).to( device=device ) # visibility at source (float)

    # apply threshold to source ADC values. returns a byte mask
    fvisi1_t  = fvisi1_t.clamp(0.0,1.0)
    fvisi2_t  = fvisi2_t.clamp(0.0,1.0)

    # make integer visi
    visi1_t = fvisi1_t.reshape( (batchsize,fvisi1_t.size()[2],fvisi1_t.size()[3]) ).long()
    visi2_t = fvisi2_t.reshape( (batchsize,fvisi2_t.size()[2],fvisi2_t.size()[3]) ).long()

    # image column origins
    meta_data_np = data["meta_%s"%(train_or_valid)].reshape((batchsize,3,4,1)).transpose((0,1,3,2))
    #print meta_data_np
    source_minx  = torch.from_numpy( meta_data_np[:,2,0,0].reshape((batchsize)) ).to(device=device)
    target1_minx = torch.from_numpy( meta_data_np[:,0,0,0].reshape((batchsize)) ).to(device=device)
    target2_minx = torch.from_numpy( meta_data_np[:,1,0,0].reshape((batchsize)) ).to(device=device)
    
    return source_t, target1_t, target2_t, flow1_t, flow2_t, visi1_t, visi2_t, fvisi1_t, fvisi2_t, source_minx, target1_minx, target2_minx,meta_data_np


## =========================================================
## DATA PREP2: Simple reader
## =========================================================

io = larcv.IOManager(larcv.IOManager.kREAD)
io.add_in_file(inputfile)
io.initialize()
def prep_data2( io, entry, batchsize, width, height, src_adc_threshold, device ):

    index = (1,0)
    source_np  = np.zeros((batchsize,1,width,height),dtype=np.float32)
    target1_np = np.zeros((batchsize,1,width,height),dtype=np.float32)
    target2_np = np.zeros((batchsize,1,width,height),dtype=np.float32)
    flow1_np   = np.zeros((batchsize,1,width,height),dtype=np.float32)
    flow2_np   = np.zeros((batchsize,1,width,height),dtype=np.float32)
    visi1_np   = np.zeros((batchsize,1,width,height),dtype=np.float32)
    visi2_np   = np.zeros((batchsize,1,width,height),dtype=np.float32)
    meta_np    = np.zeros((batchsize,3,1,4), dtype=np.float32)
    for ib in xrange(batchsize):
        io.read_entry(entry)
        
        ev_adc = io.get_data("image2d","adc")
        ev_flo = io.get_data("image2d","pixflow")
        ev_vis = io.get_data("image2d","pixvisi")

        source_np[ib,0,:,:]  = larcv.as_ndarray( ev_adc.as_vector()[2] ).transpose(1,0)
        target1_np[ib,0,:,:] = larcv.as_ndarray( ev_adc.as_vector()[0] ).transpose(1,0)
        target2_np[ib,0,:,:] = larcv.as_ndarray( ev_adc.as_vector()[1] ).transpose(1,0)

        flow1_np[ib,0,:,:] = larcv.as_ndarray( ev_flo.as_vector()[0] ).transpose(1,0)
        flow2_np[ib,0,:,:] = larcv.as_ndarray( ev_flo.as_vector()[1] ).transpose(1,0)

        visi1_np[ib,0,:,:] = larcv.as_ndarray( ev_vis.as_vector()[0] ).transpose(1,0)
        visi2_np[ib,0,:,:] = larcv.as_ndarray( ev_vis.as_vector()[1] ).transpose(1,0)

        for ip in xrange(0,3):
            meta_np[ib,ip,0,0] = ev_adc.as_vector()[ip].meta().min_x()
            meta_np[ib,ip,0,1] = ev_adc.as_vector()[ip].meta().min_y()
            meta_np[ib,ip,0,2] = ev_adc.as_vector()[ip].meta().max_x()
            meta_np[ib,ip,0,3] = ev_adc.as_vector()[ip].meta().max_y()
            
        entry += 1
        

    source_t  = torch.from_numpy( source_np ).to( device=device )   # source image ADC
    target1_t = torch.from_numpy( target1_np ) .to(device=device )   # target image ADC
    target2_t = torch.from_numpy( target2_np ) .to( device=device )  # target2 image ADC
    flow1_t   = torch.from_numpy( flow1_np ).to( device=device )   # flow from source to target
    flow2_t   = torch.from_numpy( flow2_np ).to( device=device ) # flow from source to target
    fvisi1_t  = torch.from_numpy( visi1_np ).to( device=device )  # visibility at source (float)
    fvisi2_t  = torch.from_numpy( visi2_np ).to( device=device ) # visibility at source (float)

    # apply threshold to source ADC values. returns a byte mask
    fvisi1_t  = fvisi1_t.clamp(0.0,1.0)
    fvisi2_t  = fvisi2_t.clamp(0.0,1.0)

    # make integer visi
    visi1_t = fvisi1_t.detach().reshape( (batchsize,fvisi1_t.size()[2],fvisi1_t.size()[3]) ).long()
    visi2_t = fvisi2_t.detach().reshape( (batchsize,fvisi2_t.size()[2],fvisi2_t.size()[3]) ).long()

    # image column origins
    source_minx  = torch.from_numpy( meta_np[:,2,0,0].reshape((batchsize)) ).to(device=device)
    target1_minx = torch.from_numpy( meta_np[:,0,0,0].reshape((batchsize)) ).to(device=device)
    target2_minx = torch.from_numpy( meta_np[:,1,0,0].reshape((batchsize)) ).to(device=device)

    return source_t, target1_t, target2_t, flow1_t, flow2_t, visi1_t, visi2_t, fvisi1_t, fvisi2_t, source_minx, target1_minx, target2_minx,meta_np

## =========================================================
## DATA PREP 3: LArCV2ServerWorker2
## =========================================================

def load_data( io ):
    from larcv import larcv
    import numpy as np
    
    width  = 832
    height = 512
    src_adc_threshold = 10.0

    index = (1,0)
    products = ["source","targetu","targetv","flowy2u","flowy2v","visiy2u","visiy2v","meta"]
    data = {}
    for k in products:
        if k !="meta":
            data[k] = np.zeros( (1,width,height), dtype=np.float32 )
        else:
            data[k] = np.zeros( (3,width,height), dtype=np.float32 )            
        
    ev_adc = io.get_data("image2d","adc")
    ev_flo = io.get_data("image2d","pixflow")
    ev_vis = io.get_data("image2d","pixvisi")

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

ioserver = LArCV2Server(batchsize_valid,"valid",VALID_LARCV_CONFIG,"ThreadProcessorValid",1,load_func=load_data,inputfile=inputfile,port=1)
def prep_data3( ioserver, batchsize, width, height, src_adc_threshold, device ):
    """
    inputs
    ------
    larcvloader: instance of LArCVDataloader
    train_or_valid (str): "train" or "valid"
    batchsize (int)
    width (int)
    height(int)
    src_adc_threshold (float)

    outputs
    -------
    source_var (Pytorch Variable): source ADC
    targer_var (Pytorch Variable): target ADC
    flow_var (Pytorch Variable): flow from source to target
    visi_var (Pytorch Variable): visibility of source (long)
    fvisi_var(Pytorch Variable): visibility of target (float)
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
    
    return source_t, target1_t, target2_t, flow1_t, flow2_t, visi1_t, visi2_t, fvisi1_t, fvisi2_t, source_minx, target1_minx, target2_minx,meta_data_np
    

nentries = 4
start_entry = 24

current_run    = -1
current_subrun = -1
current_event  = -1

for ientry in xrange(start_entry,start_entry+nentries,batchsize_valid):

    # canvas for debug
    c = rt.TCanvas("c","c",1500,1000)
    c.Divide(3,2)
    c.Draw()


    # get data
    #source,target1,target2,flow1,flow2,visi1,visi2,fvisi1,fvisi2,Wminx,Uminx,Vminx,srcmeta = prep_data( iovalid, "valid", batchsize_valid, 
    #                                                                                                    IMAGE_WIDTH, IMAGE_HEIGHT, ADC_THRESH, DEVICE )
    #source,target1,target2,flow1,flow2,visi1,visi2,fvisi1,fvisi2,Wminx,Uminx,Vminx,srcmeta = prep_data2( io, ientry, batchsize_valid, 
    #                                                                                                     IMAGE_WIDTH, IMAGE_HEIGHT, ADC_THRESH, DEVICE )
    source,target1,target2,flow1,flow2,visi1,visi2,fvisi1,fvisi2,Wminx,Uminx,Vminx,srcmeta = prep_data3( ioserver, batchsize_valid, 
                                                                                                         IMAGE_WIDTH, IMAGE_HEIGHT, ADC_THRESH, DEVICE )
    
    
    # make imagemeta for source
    srcmetas = []
    targetumetas = []
    targetvmetas = []    
    for b in xrange(batchsize_valid):
        meta = larcv.ImageMeta( srcmeta[b,2,0,0], srcmeta[b,2,0,1], srcmeta[b,2,0,2], srcmeta[b,2,0,3], IMAGE_HEIGHT, IMAGE_WIDTH, 2 )
        srcmetas.append(meta)
        meta = larcv.ImageMeta( srcmeta[b,0,0,0], srcmeta[b,0,0,1], srcmeta[b,0,0,2], srcmeta[b,0,0,3], IMAGE_HEIGHT, IMAGE_WIDTH, 2 )
        targetumetas.append(meta)
        meta = larcv.ImageMeta( srcmeta[b,1,0,0], srcmeta[b,1,0,1], srcmeta[b,1,0,2], srcmeta[b,1,0,3], IMAGE_HEIGHT, IMAGE_WIDTH, 2 )
        targetvmetas.append(meta)

    # check image2d input
    img_adc = larcv.as_image2d_meta( source[0,0,:,:].detach().cpu().numpy().transpose((1,0)), srcmetas[0] )
    hadc = larcv.as_th2d( img_adc, "hadc_input" )
    c.cd(1)
    hadc.SetTitle("source y")
    hadc.Draw("COLZ")
    c.Update()
    #c.SaveAs("hadc_input_%d.png"%(ientry))

    #cv.imwrite( "cvadc_input_%d.png"%(ientry), source[0,0,:,:].detach().cpu().numpy() )

    img_targetu = larcv.as_image2d_meta( target1[0,0,:,:].detach().cpu().numpy().transpose((1,0)), targetumetas[0] )
    htargetu = larcv.as_th2d( img_targetu, "htargetu_input" )
    c.cd(2)
    htargetu.SetTitle("target u")
    htargetu.Draw("COLZ")
    c.Update()
    #c.SaveAs("htargetu_input_%d.png"%(ientry))

    img_targetv = larcv.as_image2d_meta( target2[0,0,:,:].detach().cpu().numpy().transpose((1,0)), targetvmetas[0] )
    htargetv = larcv.as_th2d( img_targetv, "htargetv_input" )
    c.cd(3)
    htargetv.SetTitle("target v")
    htargetv.Draw("COLZ")
    #c.Update()
    #c.SaveAs("htargetv_input_%d.png"%(ientry))
    
    img_truey2u = larcv.as_image2d_meta( flow1[0,0,:,:].detach().clamp(-832,832).cpu().numpy().transpose((1,0)), srcmetas[0] )
    htruey2u = larcv.as_th2d( img_truey2u, "htruey2u_input" )
    c.cd(5)        
    htruey2u.SetTitle("true y2u")
    htruey2u.Draw("COLZ")
    c.Update()
    #c.SaveAs("hflowy2u_true_%d.png"%(ientry))
    #cv.imwrite( "cvflowy2u_input_%d.png"%(ientry), flow1[0,0,:,:].detach().clamp(-256,256).cpu().numpy() )
    
    img_truey2v = larcv.as_image2d_meta( flow2[0,0,:,:].detach().clamp(-832,832).cpu().numpy().transpose((1,0)), srcmetas[0] )
    htruey2v = larcv.as_th2d( img_truey2v, "htruey2v_input" )
    c.cd(6)    
    htruey2v.SetTitle("true y2v")
    htruey2v.Draw("COLZ")
    c.Update()
    #c.SaveAs("hflowy2v_true_%d.png"%(ientry))
    #cv.imwrite( "cvflowy2v_input_%d.png"%(ientry), flow2[0,0,:,:].detach().clamp(-256,256).cpu().numpy() )

    # true visi
    img_visiy2u = larcv.as_image2d_meta( fvisi1[0,0,:,:].detach().clamp(0,1.0).cpu().numpy().transpose((1,0)), srcmetas[0] )
    hvisiy2u = larcv.as_th2d( img_visiy2u, "hvisiy2u_true" )
    #hvisiy2u.Draw("COLZ")
    #c.Update()
    #c.SaveAs("hvisiy2u_true_%d.png"%(ientry))
    
    img_visiy2v = larcv.as_image2d_meta( fvisi2[0,0,:,:].detach().clamp(0.0,1.0).cpu().numpy().transpose((1,0)), srcmetas[0] )
    hvisiy2v = larcv.as_th2d( img_visiy2v, "hvisiy2v_input" )
    #hvisiy2v.Draw("COLZ")
    #c.Update()
    #c.SaveAs("hvisiy2v_true_%d.png"%(ientry))
    
    #flowy2u, flowy2v = model(source,target1,target2)
    flowy2u = flow1
    flowy2v = flow2
    print "flowshape: ",flowy2u.shape

    print "Draw prediction image"
    hpredict = hadc.Clone("hpredict")
    hpredict.Reset()
    for col in xrange(0,832):
        for r in xrange(0,512):
            #if fvisi1[0,0,col,r]<0.5:
            #    continue
            if hadc.GetBinContent(col+1,r+1)<10:
                continue
            flowcol = int(col+flowy2u[0,0,col,r])
            if flowcol>=0 and flowcol<832:
                hpredict.SetBinContent( flowcol+1, r+1, hadc.GetBinContent(col+1,r+1) )
            #hpredict.SetBinContent( col+1, r+1, flowy2u[0,0,col,r]*fvisi1[0,0,col,r] )
    #hpredict.Draw("COLZ")
    #c.SaveAs("predict_%d.png"%(ientry))

    print "Draw true image"
    htrue = hadc.Clone("htrue")
    htrue.Reset()
    for col in xrange(0,832):
        for r in xrange(0,512):
            if hadc.GetBinContent(col+1,r+1)<10:
                continue
            flowcol = int(col+flow1[0,0,col,r])
            if flowcol>=0 and flowcol<832:
                htrue.SetBinContent( flowcol+1, r+1, hadc.GetBinContent(col+1,r+1) )
            #htrue.SetBinContent( col+1,r+1, flow1[0,0,col,r] )
    #htrue.Draw("COLZ")
    #c.SaveAs("true_%d.png"%(ientry))
    
    
    aved2,posy2u,posy2v = loss(flowy2u,flowy2v,fvisi1,fvisi2, Wminx, Uminx, Vminx )
    #print "loss: ", aved2.item()
    aved = aved2.clamp(0.0,1.0e6).sqrt()
    print aved.shape
    aved_np = aved[0,:,:].detach().cpu().numpy()
    print aved_np.shape

    #print aved2_py,aved2_py*1e-3
    aved_img  = larcv.as_image2d_meta( aved_np.transpose(1,0), srcmetas[0] )
    aved_th2d = larcv.as_th2d( aved_img, "loss3d" )
    c.cd(4)
    aved_th2d.SetTitle("consistency dist")
    aved_th2d.Draw("COLZ")
    c.Update()
    #c.SaveAs("loss3d_%d.png"%(ientry))
    c.SaveAs("loss3d_check_%d.png"%(ientry))
    
    npixs = (fvisi1*fvisi2).sum()
    l2 = aved2.sum()/npixs
    print "Ave pixel loss: ",l2.item()

    del hadc
    del htargetu
    del htargetv
    del htruey2u
    del htruey2v
    del hvisiy2v
    del hvisiy2u
    del hpredict
    del htrue
    del aved_th2d
    
    print "Entry to continue."
    #raw_input()

