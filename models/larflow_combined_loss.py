import os,sys,time
from array import array

import ROOT as rt
from larcv import larcv

import numpy as np

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F

from larflow_flow_loss import LArFlowFlowLoss
from larflow_visibility_loss import LArFlowVisibilityLoss
from larflow_consistency3d_loss import LArFlow3DConsistencyLoss    

class LArFlowCombinedLoss(nn.Module):
    def __init__(self, ncols, nrows, batchsize, maxdist_err, visi_weight, consistency_weight, weight=None, size_average=True, ignore_index=-100 ):
        super(LArFlowCombinedLoss,self).__init__()
        self.ignore_index = ignore_index
        self.reduce = False

        self.visi_weight        = visi_weight
        self.consistency_weight = consistency_weight

        # flow loss
        self.flowloss1 = LArFlowFlowLoss(maxdist_err)
        self.flowloss2 = LArFlowFlowLoss(maxdist_err)
        
        # visibility parameters
        self.visiloss1 = LArFlowVisibilityLoss()
        self.visiloss2 = LArFlowVisibilityLoss()

        # consistency loss
        self.consistency3d = LArFlow3DConsistencyLoss(ncols,nrows,batchsize,goodrange=[832/2-308/2+1,832/2+310/2-1])
 
    def forward(self,flow1_predict,flow2_predict,
                visi1_predict,visi2_predict,
                flow1_truth,flow2_truth,
                visi1_truth,visi2_truth,
                source_minx, target1_minx, target2_minx):
        """
        flow_predict: (b,1,h,w) tensor with flow prediction
        flow_truth:   (b,1,h,w) tensor with correct flow value
        visi_predict: (b,1,h,w) tensor with visibility prediction. values between [0,1].
        visi_truth:   (b,h,w) tensor with correct visibility. values either 0 or 1 (long)
        fvisi_truth:  (b,1,h,w) tensor with correct visibility. values either 0 or 1 (float)
        """
            
        # flow
        floss1 = self.flowloss1(flow1_predict,flow1_truth,visi1_truth)
        if flow2_predict is not None:
            floss2 = self.flowloss2(flow2_predict,flow2_truth,visi2_truth)
        else:
            floss2 = torch.zeros( 1, dtype=torch.float ).to(device=flow1_predict.device)

        # visi
        if visi1_predict is not None and self.visi_weight>0:
            vloss1 = self.visiloss1(visi1_predict,visi1_truth)*self.visi_weight
        else:
            vloss1 = torch.zeros( 1, dtype=torch.float ).to(device=flow1_predict.device)
            
        if visi2_predict is not None and self.visi_weight>0:
            vloss2 = self.visiloss2(visi2_predict,visi2_truth)*self.visi_weight
        else:
            vloss2 = torch.zeros( 1, dtype=torch.float ).to(device=flow1_predict.device)

        # consistency
        if self.consistency_weight>0 and flow2_predict is not None:
            if visi1_truth is not None:
                fvisi1 = visi1_truth.float().clamp(0.0,1.0).reshape( flow1_predict.shape )
            else:
                fvisi1 = None
            if visi2_truth is not None:
                fvisi2 = visi2_truth.float().clamp(0.0,1.0).reshape( flow2_predict.shape )
            else:
                fvisi2 = None
            closs = self.consistency3d(flow1_predict,flow2_predict,fvisi1,fvisi2,source_minx,target1_minx,target2_minx)*self.consistency_weight
        else:
            closs = torch.zeros( 1, dtype=torch.float ).to(device=flow1_predict.device)

        loss = floss1+floss2+vloss1+vloss2+closs
        return loss,floss1,floss2,vloss1,vloss2,closs


if __name__ == "__main__":

    device = torch.device("cuda:0")
    
    lossfunc = LArFlowCombinedLoss(832,512,1,100.0,0.0,1.0)
    
    # save a histogram
    rout = rt.TFile("testout_totalloss_ub.root","recreate")
    ttest = rt.TTree("test","Consistency 3D Loss test data")
    dloss = array('d',[0])
    dtime = array('d',[0])
    dback = array('d',[0])    
    dfloss1 = array('d',[0])
    dfloss2 = array('d',[0])
    dvloss1 = array('d',[0])
    dvloss2 = array('d',[0])
    dvloss2 = array('d',[0])
    dcloss  = array('d',[0])
    ttest.Branch("dtime",dtime,"dtime/D")
    ttest.Branch("dback",dback,"dback/D")    
    ttest.Branch("loss",dloss,"loss/D")
    ttest.Branch("floss1",dfloss1,"floss1/D")
    ttest.Branch("floss2",dfloss2,"floss2/D")
    ttest.Branch("vloss1",dvloss1,"vloss1/D")
    ttest.Branch("vloss2",dvloss2,"vloss2/D")
    ttest.Branch("closs",dcloss,"closs/D")    

    # as test, we process some pre-cropped small samples
    io = larcv.IOManager()
    io.add_in_file( "../testdata/smallsample/larcv_dlcosmictag_5482426_95_smallsample082918.root" ) # create a unit test file (csv)
    io.initialize()

    nentries = io.get_n_entries()
    print "Number of Entries: ",nentries
    start = time.time()

    istart=0
    iend=nentries
    #istart=155
    #iend=1
    
    for ientry in xrange(istart,iend):

        tentry = time.time()
        
        io.read_entry( ientry )
        if os.environ["LARCV_VERSION"]=="1":
            ev_adc_test = io.get_data(larcv.kProductImage2D,"adc")            
            ev_flowy2u_test = io.get_data(larcv.kProductImage2D,"larflow_y2u")
            ev_flowy2v_test = io.get_data(larcv.kProductImage2D,"larflow_y2v")
            ev_trueflow_test = io.get_data(larcv.kProductImage2D,"pixflow")
            ev_truevisi_test = io.get_data(larcv.kProductImage2D,"pixvisi")
            flowy2u = ev_flowy2u_test.Image2DArray()[0]
            flowy2v = ev_flowy2v_test.Image2DArray()[0]
            truey2u = ev_trueflow_test.Image2DArray()[0]
            truey2v = ev_trueflow_test.Image2DArray()[1] 
            visiy2u = ev_truevisi_test.Image2DArray()[0]
            visiy2v = ev_truevisi_test.Image2DArray()[1]
            source_meta  = ev_adc_test.Image2DArray()[2].meta()
            targetu_meta = ev_adc_test.Image2DArray()[0].meta()
            targetv_meta = ev_adc_test.Image2DArray()[1].meta()
            
        elif os.environ["LARCV_VERSION"]=="2":
            ev_adc_test = io.get_data("image2d","adc")            
            ev_flowy2u_test = io.get_data("image2d","larflow_y2u")
            ev_flowy2v_test = io.get_data("image2d","larflow_y2v")
            ev_trueflow_test = io.get_data("image2d","pixflow")
            ev_truevisi_test = io.get_data("image2d","pixvisi")

            flowy2u = ev_flowy2u_test.as_vector()[0]
            flowy2v = ev_flowy2v_test.as_vector()[0]
            truey2u = ev_trueflow_test.as_vector()[0]
            truey2v = ev_trueflow_test.as_vector()[1] 
            visiy2u = ev_truevisi_test.as_vector()[0]
            visiy2v = ev_truevisi_test.as_vector()[1]
            source_meta  = ev_adc_test.as_vector()[2].meta()
            targetu_meta = ev_adc_test.as_vector()[0].meta()
            targetv_meta = ev_adc_test.as_vector()[1].meta()
        

        # numpy arrays
        index = (0,1)
        if os.environ["LARCV_VERSION"]=="2":
            index = (1,0)
        
        np_flowy2u = larcv.as_ndarray(flowy2u).transpose(index).reshape((1,1,source_meta.cols(),source_meta.rows()))
        np_flowy2v = larcv.as_ndarray(flowy2v).transpose(index).reshape((1,1,source_meta.cols(),source_meta.rows()))
        np_visiy2u = larcv.as_ndarray(visiy2u).transpose(index).reshape((1,1,source_meta.cols(),source_meta.rows()))
        np_visiy2v = larcv.as_ndarray(visiy2v).transpose(index).reshape((1,1,source_meta.cols(),source_meta.rows()))
        np_trueflowy2u = larcv.as_ndarray(truey2u).transpose(index).reshape((1,1,source_meta.cols(),source_meta.rows()))
        np_trueflowy2v = larcv.as_ndarray(truey2v).transpose(index).reshape((1,1,source_meta.cols(),source_meta.rows()))        
        
        # tensor conversion
        predflow_y2u_t = torch.from_numpy( np_flowy2u ).to(device=device)
        predflow_y2v_t = torch.from_numpy( np_flowy2v ).to(device=device)

        trueflow_y2u_t = torch.from_numpy( np_trueflowy2u ).to(device=device)
        trueflow_y2v_t = torch.from_numpy( np_trueflowy2v ).to(device=device)

        truevisi_y2u_t = torch.from_numpy( np_visiy2u ).to(device=device)
        truevisi_y2v_t = torch.from_numpy( np_visiy2v ).to(device=device)

        # pick the input
        y2u_t = predflow_y2u_t.requires_grad_()
        y2v_t = predflow_y2v_t.requires_grad_()
        #y2u_t = trueflow_y2u_t.clone().requires_grad_()
        #y2v_t = trueflow_y2v_t.clone().requires_grad_()

        # calculte the loss
        totloss,floss1,floss2,vloss1,vloss2,closs = lossfunc(y2u_t,y2v_t,
                                                             None,None,
                                                             trueflow_y2u_t,trueflow_y2v_t,
                                                             truevisi_y2u_t,truevisi_y2v_t,
                                                             source_meta.min_x(),targetu_meta.min_x(),targetv_meta.min_x())
        # forward timing
        dtime[0] = time.time()-tentry
        # backward test
        tback = time.time()
        totloss.backward()
        dback[0] = time.time()-tback
        print "  runbackward: ",time.time()-tback," secs"
        
        print "Loss (iter {}): {}".format(ientry,totloss.item())," iscuda",totloss.is_cuda
        dloss[0] = totloss.item()
        dfloss1[0] = floss1.item()
        dfloss2[0] = floss2.item()
        dvloss1[0] = vloss1.item()
        dvloss2[0] = vloss2.item()        
        dcloss[0]  = closs.item()
        
        ttest.Fill()
        
    end = time.time()
    tloss = end-start
    print "Time: ",tloss," secs / ",tloss/nentries," secs per event"
    rout.cd()
    ttest.Write()
    rout.Close()


