import os,sys,time
from array import array

import numpy as np

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F

# ROOT
import ROOT as rt
from larcv import larcv

# taken from torch.nn.modules.loss
def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as not requiring gradients"

class LArFlow3DConsistencyLoss:
    def __init__(self,ncols, nrows, intersectiondata=None, nsource_wires=3456, ntarget_wires=2400):
        if intersectiondata is None:
            # set default
            if os.environ["LARCV_VERSION"].strip()=="1":
                intersectiondata = "../gen3dconsistdata/consistency3d_data_larcv1.root"
            elif os.environ["LARCV_VERSION"].strip()=="2":
                intersectiondata = "../gen3dconsistdata/consistency3d_data_larcv2.root"
            else:
                raise RuntimeError("Invalid LARCV_VERSION: {}".format(LARCV_VERSION))

        if not os.path.exists(intersectiondata):
            raise RuntimeError("could not find intersection data file: {}".format(intersectiondata))
            
        self.nsource_wires = nsource_wires
        self.ntarget_wires = ntarget_wires        
        self._load_data(ncols, nrows, intersectiondata)
        
    def _load_data(self, ncols, nrows, intersectiondata):
        # intersection location (y,z) for (source,target) intersections
        self.intersections_t  = torch.zeros( (2, 2, self.nsource_wires, self.ntarget_wires ) ).float()

        # fill intersection matrix (should make image2d instead of this loop fill
        io = larcv.IOManager()
        if os.environ["LARCV_VERSION"]=="1":
            io.add_in_file(intersectiondata)
            io.initialize()            
            ev_y2u = io.get_data(larcv.kProductImage2D,"y2u_intersect")
            if ev_y2u.Image2DArray().size()!=2:
                raise RuntimeError("Y2U intersection image2d vector should be len 2 (for detector y,z)")
            self.intersections_t[0,0,:,:] = torch.from_numpy( larcv.as_ndarray( ev_y2u.Image2DArray()[0] ).reshape(self.ntarget_wires,self.nsource_wires).transpose((1,0)) )
            self.intersections_t[0,1,:,:] = torch.from_numpy( larcv.as_ndarray( ev_y2u.Image2DArray()[1] ).reshape(self.ntarget_wires,self.nsource_wires).transpose((1,0)) )
            ev_y2v = io.get_data(larcv.kProductImage2D,"y2v_intersect")
            if ev_y2v.Image2DArray().size()!=2:
                raise RuntimeError("Y2V intersection image2d vector should be len 2 (for detector y,z)")
            self.intersections_t[1,0,:,:] = torch.from_numpy( larcv.as_ndarray( ev_y2v.Image2DArray()[0] ).reshape(self.ntarget_wires,self.nsource_wires).transpose((1,0)) )
            self.intersections_t[1,1,:,:] = torch.from_numpy( larcv.as_ndarray( ev_y2v.Image2DArray()[1] ).reshape(self.ntarget_wires,self.nsource_wires).transpose((1,0)) )
        elif os.environ["LARCV_VERSION"]=="2":
            io.add_in_file(intersectiondata)
            io.initialize()
            ev_y2u = io.get_data("image2d","y2u_intersect")
            ev_y2v = io.get_data("image2d","y2v_intersect")
            self.intersections_t[0,0,:,:] = torch.from_numpy( larcv.as_ndarray( ev_y2u.as_vector()[0] ).transpose((1,0)) )
            self.intersections_t[0,1,:,:] = torch.from_numpy( larcv.as_ndarray( ev_y2u.as_vector()[1] ).transpose((1,0)) )
            self.intersections_t[1,0,:,:] = torch.from_numpy( larcv.as_ndarray( ev_y2v.as_vector()[0] ).transpose((1,0)) )
            self.intersections_t[1,1,:,:] = torch.from_numpy( larcv.as_ndarray( ev_y2v.as_vector()[1] ).transpose((1,0)) )
        
        # index of source matrix: each column gets value same as index
        self.ncols = ncols
        self.nrows = nrows                        
        self.src_index_np = np.tile( np.linspace( 0, float(ncols)-1, ncols ), nrows )
        self.src_index_np = self.src_index_np.reshape( (nrows, ncols) ).transpose( (1,0) )
        self.src_index_t  = torch.from_numpy( self.src_index_np ).float()
        #print "src_index_np: ",self.src_index_np.shape#, self.src_index_np[3,:]

        # we build in offset for intersections_t. each row gets 2400*row added to it
        self.row_index_offset = np.zeros( (ncols,nrows), dtype=np.float32 )
        for i in xrange(nrows):
            self.row_index_offset[i,:]  = i*self.ntarget_wires # the column tells me the y-wire, we originate from
        self.row_index_offset_t = torch.from_numpy( self.row_index_offset )
        #print "row index offset: ",self.row_index_offset.shape#," ",self.row_index_offset[3,:]
        
    def to(self,device=torch.device("cpu")):
        self.intersections_t = self.intersections_t.to( device=device )
        self.src_index_t     = self.src_index_t.to( device=device )

    def cuda(self,gpuid):
        self.use_cuda = True
        self.gpuid = gpuid
        device = torch.device("cuda:%d"%(gpuid))
        self.intersections_t = self.intersections_t.to( device=device )
        self.src_index_t     = self.src_index_t.to( device=device )
        
    def calc_loss(self,source_meta, targetu_meta, targetv_meta,
                  flow1_predict,flow2_predict,
                  flow1_truth,flow2_truth,
                  visi1_truth,visi2_truth,
                  fvisi1_truth,fvisi2_truth):
                  
        """
        input
        -----
        flow[x]_predict: output prediction for flow with {x:0=Y2U, 1=Y2V}
        visi[x]_predict: output prediction for visibility with {x:0=Y2U, 1=Y2V}
        """
        ## debug
        #print "flow predict: ",flow1_predict.shape, flow2_predict.shape, flow1_predict.device, flow2_predict.device
        #print "flow truth: ",flow1_truth.shape, flow2_truth.shape
        #print "visi truth: ",visi1_truth.shape, visi2_truth.shape
        #print "src_index_t: ",self.src_index_t.shape, self.src_index_t.device
        #print "src_origin: ",source_meta.min_x()
        #print "tar1_origin: ",targetu_meta.min_x()
        #print "tar1_origin: ",targetv_meta.min_x()        
        
        ## algo
        ## turn some numbers into tensor
        src_min_x_t  = (torch.ones( (1), dtype=torch.float32 )*source_meta.min_x()).to( device=flow1_predict.device )
        tar1_min_x_t = (torch.ones( (1), dtype=torch.float32 )*targetu_meta.min_x()).to( device=flow1_predict.device )
        tar2_min_x_t = (torch.ones( (1), dtype=torch.float32 )*targetv_meta.min_x()).to( device=flow1_predict.device )
        
        ## we need to get source and target wire (from flow)
        source_fwire_t  = self.src_index_t.add( src_min_x_t )

        # dualflowmask: we only consider source pixels where visibility is in both (these are binary values)
        mask = fvisi1_truth.clamp(0.0,1.0)*fvisi2_truth.clamp(0.0,1.0)

        # calculate the target wire we've flowed to
        #print "flow1_predict: ",flow1_predict[340:350,91]
        pred_target1_fwire_t   = (self.src_index_t + flow1_predict).add( tar1_min_x_t ) # should be whole-image u-wire number
        #print "flow1_predict+col_index+col_origin({}): ".format(source_meta.min_x()),pred_target1_fwire_t[340:350,91]
        pred_target1_fwire_t   = pred_target1_fwire_t.round() # round to nearest wire
        #print "flow1_predict+source_origin(round): ",pred_target1_fwire_t[490:510,470]
        pred_target1_iwire_t   = pred_target1_fwire_t.long().clamp(0,2399) # cast to integer
        #print "flow1_predict+source_origin(round+clamp): ",pred_target1_iwire_t[340:350,91]
        pred_target1_index_t   = (source_fwire_t.long())*2400 + pred_target1_iwire_t # ywire*2400 + uwire

        #print "flow2_predict: ",flow2_predict[340:350,91]
        pred_target2_fwire_t   = (self.src_index_t + flow2_predict).add( tar2_min_x_t )
        
        pred_target2_fwire_t   = pred_target2_fwire_t.round() # round to nearest wire
        pred_target2_iwire_t   = pred_target2_fwire_t.long().clamp(0,2399) # cast to integer
        #print "flow2_predict+source_origin(round+clamp): ",pred_target2_iwire_t[340:350,91]
        pred_target2_index_t   = (source_fwire_t.long())*2400 + pred_target2_iwire_t # ywire*2400 + vwire

        
        # get the (y,z) of the intersection we've flowed to
        posyz_target1_t = torch.zeros( (2,self.ncols, self.nrows) ).to( device=flow1_predict.device )
        posyz_target1_t[0,:,:] = torch.take( self.intersections_t[0,0,:], pred_target1_index_t.reshape( self.ncols*self.nrows ) ).reshape( (self.ncols,self.nrows) ) # det-y
        posyz_target1_t[1,:,:] = torch.take( self.intersections_t[0,1,:], pred_target1_index_t.reshape( self.ncols*self.nrows ) ).reshape( (self.ncols,self.nrows) ) # det-y        
        posyz_target1_t[0,:,:] *= mask
        posyz_target1_t[1,:,:] *= mask

        posyz_target2_t = torch.zeros( (2,self.ncols, self.nrows) ).to( device=flow1_predict.device )
        posyz_target2_t[0,:,:] = torch.take( self.intersections_t[1,0,:], pred_target2_index_t.reshape( self.ncols*self.nrows ) ).reshape( (self.ncols,self.nrows) ) # det-y
        posyz_target2_t[1,:,:] = torch.take( self.intersections_t[1,1,:], pred_target2_index_t.reshape( self.ncols*self.nrows ) ).reshape( (self.ncols,self.nrows) ) # det-y                
        posyz_target2_t[0,:,:] *= mask
        posyz_target2_t[1,:,:] *= mask

        # calculate the squared difference between the points
        diff_yz = posyz_target1_t-posyz_target2_t # take diff
        diff_yz = diff_yz*diff_yz # square

        # loss is the mean loss per non-masked pixel
        diff_yz /= mask.sum() # divide by number of non-masked pixels
        loss = diff_yz.sum()  # calculate average loss, which is just the squared distance

        return loss

if __name__=="__main__":
    """ Code for testing"""

    losscalc = LArFlow3DConsistencyLoss(832,512)

    # set device
    device = torch.device("cuda:0")
    #device = torch.device("cpu")
    losscalc.to( device=device )    

    print "loss calculor loaded"

    # save a histogram
    rout = rt.TFile("testout_consistency3dloss.root","recreate")
    ttest = rt.TTree("test","Consistency 3D Loss test data")
    dloss = array('d',[0])
    dtime = array('d',[0])
    ttest.Branch("loss",dloss,"loss/D")
    ttest.Branch("dtime",dtime,"dtime/D")    

    # as test, we process some pre-cropped small samples
    io = larcv.IOManager()
    io.add_in_file( "../testdata/smallsample/larcv_dlcosmictag_5482426_95_smallsample082918.root" )
    io.initialize()

    nentries = io.get_n_entries()
    print "Number of Entries: ",nentries
    start = time.time()
    
    for ientry in xrange(nentries):

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
        

        # tensor conversion
        index = (0,1)
        if os.environ["LARCV_VERSION"]=="2":
            index = (1,0)
        predflow_y2u_t = torch.from_numpy( larcv.as_ndarray(flowy2u).transpose(index) ).to(device=device)
        predflow_y2v_t = torch.from_numpy( larcv.as_ndarray(flowy2v).transpose(index) ).to(device=device)

        trueflow_y2u_t = torch.from_numpy( larcv.as_ndarray(truey2u).transpose(index) ).to(device=device)
        trueflow_y2v_t = torch.from_numpy( larcv.as_ndarray(truey2v).transpose(index) ).to(device=device)

        truevisi_y2u_t = torch.from_numpy( larcv.as_ndarray(visiy2u).transpose(index) ).to(device=device)
        truevisi_y2v_t = torch.from_numpy( larcv.as_ndarray(visiy2v).transpose(index) ).to(device=device)

        #print "source meta: ",source_meta.dump()

        lossval = losscalc.calc_loss( source_meta, targetu_meta, targetv_meta,
                                      predflow_y2u_t, predflow_y2v_t, # for debugging
                                      #trueflow_y2u_t, trueflow_y2v_t, # for debug test
                                      trueflow_y2u_t, trueflow_y2v_t,
                                      truevisi_y2u_t.long(), truevisi_y2v_t.long(),
                                      truevisi_y2u_t, truevisi_y2v_t )
        print "Loss (iter {}): {}".format(ientry,lossval.item())," iscuda",lossval.is_cuda
        dloss[0] = lossval.item()
        dtime[0] = time.time()-tentry
        ttest.Fill()
        
    end = time.time()
    tloss = end-start
    print "Time: ",tloss," secs / ",tloss/nentries," secs per event"
    rout.cd()
    ttest.Write()
    rout.Close()
