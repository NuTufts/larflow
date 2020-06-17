import os,sys,time
from array import array

import numpy as np

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F

# sparse submanifold convnet library
import sparseconvnet as scn

# ROOT
import ROOT as rt
from larcv import larcv

from func_intersect_ub import IntersectUB

class SparseLArFlow3DConsistencyLoss(nn.Module):
    """
    Loss for LArFlow which combines L2 loss between predicted and true flow,
    but also an additional loss from 3D inconsistency in the Y2U and Y2V flows.

    The consistency loss is measured by taking the L2 distance between the
    3D point predicted by the Y2U and Y2V flows. Because we assume
    same tick correspondences only (fixing x) and flow from the Y plane (fixing z),
    we really are only measuring the distance in the y-detector dimension.
    """
    def __init__(self, nrows, ncols, calc_consistency=True,
                 intersectiondata=None, larcv_version=None,
                 nsource_wires=3456, ntarget_wires=2400, goodrange=None,
                 predict_classvec=False,
                 return_pos_images=False, reduce=True):
        super(SparseLArFlow3DConsistencyLoss,self).__init__()
        """
        inputs
        ------
        nrows [int]: image rows
        ncols [int]" image cols
        calc_consistency [bool]: if true, calculate loss based on flow 3d consistency
        intersectiondata [str]: path to rootfile which stores a table
                                for what Y-position corresponds to some wire crossing.
                                if none, only the flow prediction loss is used
        larcv_version [None or int]: 1 or 2 for larcv 1 or 2
        nsource_wires [int]: don't remember
        ntarget_wires [int]: don't remember
        goodrange [tuple int]: evaluate loss only in the column range with overlap
        predict_classvec [bool] if true, use softmax loss in place of regression loss
        return_pos_images [bool] don't remember
        reduct [bool]  if true, reduce loss into single number. for training, should be true.
        """

        # self-consistency data
        self.calc_consistency = calc_consistency
        if self.calc_consistency:
            IntersectUB.load_intersection_data(intersectiondatafile,larcv_version=larcv_version,
                                               nsource_wires=nsource_wires,ntarget_wires=ntarget_wires)
            IntersectUB.set_img_dims( nrows, ncols )

        if goodrange is not None:
            self.goodrange_t = torch.zeros( (ncols,nrows), dtype=torch.float )
            self.goodrange_t[goodrange[0]:goodrange[1],:] = 1.0
        else:
            self.goodrange_t = None
        self._return_pos_images = return_pos_images
        self._reduce = reduce
        self._predict_classvec = predict_classvec

        
        self.outlayer1 = scn.OutputLayer(2)
        self.outlayer2 = scn.OutputLayer(2)
        
        if not self._predict_classvec:
            # regression loss
            self.l1loss = nn.SmoothL1Loss(reduction='none')
        else:
            self.colnum = torch.from_numpy( np.arange(ncols,dtype=np.int)  ).requires_grad_(False)
            self.l1loss = nn.CrossEntropyLoss()


    def forward(self,coord, flow1_predict,flow2_predict,
                flow1_truth,flow2_truth,
                mask1_truth=None,mask2_truth=None,
                source_originx=0, targetu_originx=0, targetv_originx=0):

        """
        input
        -----
        coord_t       [SparseConvTensor (N,3)]: list of (row,col,batch)
        flow1_predict [SparseConvTensor (N,1)]: predicted flow for Y2U.
        flow2_predict [SparseConvTensor (N,1)]: predicted flow for Y2V. If None, only one-flow calculated
        flow1_truth   [tensor (N,1)]: predicted flow for Y2U.
        flow2_truth   [tensor (N,1)]: predicted flow for Y2V. Coordinates is where we is.
        """

        if self.calc_consistency:
            posyz_target1_t,posyz_target2_t = \
            IntersectUB.apply( flow1_predict, flow2_predict,
                            source_originx, targetu_originx, targetv_originx )

        # turn off operation tracking for truth tensors
        flow1_truth.requires_grad_(False)
        flow2_truth.requires_grad_(False)

        if flow1_predict is not None:
            # convert sparseconvtensor into torch.tensor in sparse representation
            flowout1 = self.outlayer1(flow1_predict)
            if mask1_truth is None:
                print "allocate mem for mask1"                
                mask1 = torch.ones( flow1_truth.shape, dtype=torch.float, requires_grad=False ).to(flow1_truth.device)
                mask1[ torch.eq(flow1_truth,-4000.0) ] = 0                
            else:
                mask1 = mask1_truth
            #print "mask1 counts: ",mask1.shape,"raw sum=",mask1.detach().sum()        
            #print "mask1 counts: ",mask1.shape,"select sum=",mask1.detach().sum()
            n1 = mask1.sum()

            if not self._predict_classvec:
                # regression loss
                flow1err = 0.1*self.l1loss(10.0*flow1_truth,10.0*flowout1)*mask1
                flow1loss = flow1err.sum()
                if n1>0:
                    flow1loss = flow1loss/n1
            else:
                # classification loss
                # first, we need to true flow prediction to source column number to get true target column
                #print "coord=",coord[:,1].shape," flow1_truth",flow1_truth[:,0].shape," mask1=",mask1[:,0].shape
                flow1target = flow1_truth[:,0].type(torch.long)*mask1[:,0].type(torch.long)
                
                # calculate cross entropy loss and weight by mask
                #print " flowout1=",flowout1.shape," flowtarget=",flow1target.shape
                flow1err  = F.cross_entropy( flowout1, flow1target, reduction='none' )*mask1[:,0]
                #print "classvec-flow1err: ",flow1err.shape
                
                # sum
                flow1loss = flow1err.sum()
                # take mean by dividing good flow points (if > 0 )
                if n1>0:
                    flow1loss = flow1loss/n1

                #print "classvec-flow1loss: ",flow1loss
                
        else:
            flow1loss = None
        
        if flow2_predict is not None:
            # convert sparseconv tensor into torch.tensor in sparse representation
            flowout2 = self.outlayer2(flow2_predict)

            if mask2_truth is None:
                print "allocate mem for mask2"
                mask2 = torch.ones( flow2_truth.shape, dtype=torch.float, requires_grad=False ).to(flow2_truth.device)
                #print "mask2 counts: ",mask2.shape,"raw sum=",mask2.detach().sum()            
                mask2[ torch.eq(flow2_truth,-4000.0) ] = 0            
                #print "mask2 counts: ",mask2.shape,"select sum=",mask2.detach().sum()
            else:
                mask2 = mask2_truth
                
            n2 = mask2.sum()

            if not self._predict_classvec:
                # regression loss
                flow2err = 0.1*self.l1loss(10.0*flow2_truth,10.0*flowout2)*mask2
                flow2loss = flow2err.sum()
                if n2>0:
                    flow2loss = flow2loss/n2
            else:
                # classification loss
                # first, we need to true flow prediction to source column number to get true target column
                flow2target = flow2_truth[:,0].type(torch.long)*mask2[:,0].type(torch.long)
                # calculate cross entropy loss and weight by mask
                flow2err  = F.cross_entropy( flowout2, flow2target, reduction='none' )*mask2[:,0]
                #print "classvec-flow2err: ",flow2err.shape                
                # sum
                flow2loss = flow2err.sum()
                # take mean by dividing good flow points (if > 0 )
                if n2>0:
                    flow2loss = flow2loss/n2
                #print "classvec-flow2loss: ",flow2loss
        else:
            flow2loss = None

        #print "flow1_truth: ",flow1_truth[0:20,0]
        #print "mask1_truth: ",mask1[0:20,0]

        #print posyz_target1_t.size()," vs. mask=",mask.size()
        if self.calc_consistency:
            posyz_target1_t *= mask1
            posyz_target2_t *= mask2

        # flow prediction loss
        
        #flow1err = (flow1_truth-flowout1)*mask1
        #flow2err = (flow2_truth-flowout2)*mask2
        #flow1err = self.l1loss(flow1_truth,flowout1)*mask1
        #flow1err = 10.0*self.l1loss(0.1*flow1_truth,0.1*flowout1)*mask1
        #flow1err[ flow1err!=flow1err ] = 0.0
        #torch.clamp( flow1err, 0, 1000 )
        #flow1loss = flow1err.sum()
        #if n1>0:
        #    flow1loss = flow1loss/n1
        
        # if flow2_predict is not None:
        #     flow2err = self.l1loss(flow2_truth,flowout2)*mask2
        #     flow2err = 0.1*self.l1loss(10.0*flow2_truth,10.0*flowout2)*mask2
        #     #flow2err[ flow2err!=flow2err ] = 0.0
        #     #torch.clamp( flow2err, 0, 1000  )

        #     flow2loss = flow2err.sum()
        #     if n2>0:
        #         flow2loss = flow2loss/n2
        # else:
        #     flow2loss = None
            
        #print "posyz 1: ",np.argwhere( np.isnan( posyz_target1_t.detach().cpu().numpy() ) )
        #print "posyz 2: ",np.argwhere( np.isnan( posyz_target2_t.detach().cpu().numpy() ) )

        # calculate the squared difference between the points
        #diff_yz = posyz_target1_t-posyz_target2_t # take diff
        #l2 = diff_yz[:,0,:,:]*diff_yz[:,0,:,:] + diff_yz[:,1,:,:]*diff_yz[:,1,:,:] # square

        #print "diffyz: ",np.argwhere( np.isnan( diff_yz.detach().cpu().numpy() ) )
        #print "mask.sum: ",np.argwhere( np.isnan( mask.sum().detach().cpu().numpy() ) )
        if self._reduce:
            if flow2_predict is not None and flow1_predict is not None:
                # add two flow errors together, weighting by mask sum
                l2flow = 0.5*(flow1loss + flow2loss)
            elif flow1_predict is not None:
                l2flow = flow1loss
            elif flow2_predict is not None:
                l2flow = flow2loss
        else:
            if flow1_predict is not None and flow2_predict is not None:
                l2flow = 0.5*(flow1err+flow2err)
            elif flow1_predict is not None:
                l2flow = flow1err
            elif flow2_predict is not None:
                l2flow = flow2err
                
        if flow1loss is not None and flow1loss!=flow1loss:
            print "NAN HAPPENED"
            print "flow1: ",(flowout1!=flowout1).sum()
            print "truth1: ",(flow1_truth!=flow1_truth).sum()
            print "flowerr1: ",(flow1err!=flow1err).sum()
            print "mask1: ",mask1.sum()
        if flow2loss is not None and flow2loss!=flow2loss:
            print "NAN HAPPENED"
            print "flow2: ",(flowout2!=flowout2).sum()
            print "truth2: ",(flow2_truth!=flow2_truth).sum()
            print "flowerr2: ",(flow2err!=flow2err).sum()
            print "mask2: ",mask2.sum()
            
        return l2flow,flow1loss,flow2loss

    
if __name__ == "__main__":

    from sparselarflowdata import load_larflow_larcvdata
    from loss_sparse_larflow import SparseLArFlow3DConsistencyLoss
    
    nrows = 512
    ncols = 832
    loss_w_3dconsistency = SparseLArFlow3DConsistencyLoss( nrows, ncols,
                                                           calc_consistency=False,
                                                           larcv_version=1,
                                                           predict_classvec=True )

    # test data
    inputfile = "/home/twongj01/data/larflow_sparse_training_data/larflow_sparsify_cropped_valid_v5.root"
    batchsize = 1
    nworkers  = 1
    flowdirs = ['y2u','y2v']
    TICKBACKWARD=False
    io = load_larflow_larcvdata( "test", [inputfile],
                                 batchsize, nworkers,
                                 producer_name="sparsecropdual",
                                 nflows=len(flowdirs),
                                 tickbackward=TICKBACKWARD,
                                 readonly_products=None )

    tstart = time.time()

    nentries = 1
    
    for n in xrange(nentries):
        flowdict = io.get_tensor_batch(torch.device("cpu"))

        coord_t  = flowdict["coord"]
        srcpix_t = flowdict["src"]
        tarpix_flow1_t = flowdict["tar1"]
        tarpix_flow2_t = flowdict["tar2"]
        truth_flow1_t  = flowdict["flow1"]
        truth_flow2_t  = flowdict["flow2"]
        # masks
        if "mask1" in flowdict:
            mask1_t = flowdict["mask1"]
        else:
            mask1_t = None
            
        if "mask2" in flowdict:
            mask2_t = flowdict["mask2"]
        else:
            mask2_t = None

        # class predicetion vec, need to fake output
        print srcpix_t.shape[0]
        out1_np = np.ones( (srcpix_t.shape[0],ncols), dtype=np.float32 )*-1.0e2
        out2_np = np.ones( (srcpix_t.shape[0],ncols), dtype=np.float32 )*-1.0e2
        oob1 = 0
        oob2 = 0
        for idx in range(srcpix_t.shape[0]):
            
            if truth_flow1_t[idx]>=0 and truth_flow1_t[idx]<ncols:
                out1_np[idx, int(truth_flow1_t[idx]) ] = 1.0
                
            if truth_flow2_t[idx]>=0 and truth_flow2_t[idx]<ncols:
                out2_np[idx, int(truth_flow2_t[idx]) ] = 1.0
                    
            if truth_flow1_t[idx]>=ncols:
                oob1 += 1
            if truth_flow2_t[idx]>=ncols:
                oob2 += 1
            if truth_flow1_t[idx]<0 and mask1_t[idx]>0:
                oob1 += 1
            if truth_flow2_t[idx]<0 and mask2_t[idx]>0:
                oob2 += 1

        print "out of bounds: flow1=",oob1," flow2=",oob2
        for idx in range(srcpix_t.shape[0]):
            if truth_flow1_t[idx]>=0:
                print idx,": ",np.argmax( out1_np[idx,:] ), truth_flow1_t[idx]
            
        flow1 = scn.InputLayer( 2, (nrows,ncols), mode=0 )( (coord_t, torch.from_numpy(out1_np), 1 ) )
        flow2 = scn.InputLayer( 2, (nrows,ncols), mode=0 )( (coord_t, torch.from_numpy(out2_np), 1 ) )
        print mask1_t.sum(),mask2_t.sum()

        totloss, flow1loss, flow2loss = loss_w_3dconsistency( coord_t,
                                                              flow1, flow2,
                                                              truth_flow1_t, truth_flow2_t,
                                                              mask1_truth=mask1_t, mask2_truth=mask2_t  )

        print totloss,flow1loss,flow2loss

        # accuracy check
        col_predicted = torch.argmax( torch.from_numpy(out1_np), 1 ).type(torch.float)
        #flow_err = ( col_predicted - flow_truth[:,0] )*mask        
        col_predicted -= truth_flow1_t[:,0]
        col_predicted *= mask1_t[:,0]
        flow_err = col_predicted.abs()
        print flow_err.sum()
        
                              
        
