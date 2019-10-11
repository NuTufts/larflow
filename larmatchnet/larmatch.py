from collections import OrderedDict
import torch
import torch.nn as nn
import sparseconvnet as scn
from utils_sparselarflow import create_resnet_layer

class LArMatch(nn.Module):

    def __init__(self,ndimensions=2,inputshape=(3456,1028),
                 nlayers=8,features_per_layer=16,
                 input_nfeatures=1,
                 stem_nfeatures=32,
                 classifier_nfeatures=[32,32],
                 leakiness=0.1,
                 neval=20000,
                 device=torch.device("cpu")):
        super(LArMatch,self).__init__()

        self.source_inputlayer  = scn.InputLayer(ndimensions,inputshape,mode=0)
        self.target1_inputlayer = scn.InputLayer(ndimensions,inputshape,mode=0)
        self.target2_inputlayer = scn.InputLayer(ndimensions,inputshape,mode=0)

        # STEM
        self.stem = scn.Sequential() 
        self.stem.add( scn.SubmanifoldConvolution(ndimensions, input_nfeatures, stem_nfeatures, 3, False ) )
        self.stem.add( scn.BatchNormLeakyReLU(stem_nfeatures,leakiness=leakiness) )

        # RESNET BLOCK
        self.resnet_layers = create_resnet_layer(10, stem_nfeatures, features_per_layer, leakiness=leakiness )

        # OUTPUT FEATURES
        self.nfeatures = features_per_layer
        self.feature_layer = scn.SubmanifoldConvolution(ndimensions, features_per_layer, self.nfeatures, 1, True )

        # from there, we move back into a tensor
        self.source_outlayer  = scn.OutputLayer(ndimensions)
        self.target1_outlayer = scn.OutputLayer(ndimensions)
        self.target2_outlayer = scn.OutputLayer(ndimensions)

        # CLASSIFER: MATCH/NO-MATCH
        classifier_layers = OrderedDict()
        classifier_layers["class0conv"] = torch.nn.Conv1d(2*features_per_layer,classifier_nfeatures[0],1)
        classifier_layers["class0relu"] = torch.nn.ReLU()
        for ilayer,nfeats in enumerate(classifier_nfeatures[1:]):
            classifier_layers["class%dconv"%(ilayer+1)] = torch.nn.Conv1d(nfeats,nfeats,1)
            classifier_layers["class%drelu"%(ilayer+1)] = torch.nn.ReLU()
        classifier_layers["classout"] = torch.nn.Conv1d(nfeats,1,1)
        self.classifier = torch.nn.Sequential( classifier_layers )


        # POINTS TO EVAL PER IMAGE
        self.neval = neval

    def forward( self, coord_src_t, src_feat_t,
                 coord_tar1_t, tar1_feat_t,
                 coord_tar2_t, tar2_feat_t,
                 pair_flow1_v, pair_flow2_v, batchsize,
                 DEVICE, return_truth=False,
                 npts1=None, npts2=None ):
        """
        run the network

        inputs
        ------
        coord_flow1_t [ (N,3) Torch Tensor ]: list of (row,col,batchid) N pix coordinates
        src_feat_t  [ (N,) torch tensor ]: list of pixel values for source image
        tar1_feat_t [ (N,) torch tensor ]: list of pixel values for target 1 image
        tar2_feat_t [ (N,) torch tensor ]: list of pixel values for target 2 image. provide NULL if nflows==1
        batchsize [int]: batch size

        outputs
        -------
        [ (N,) torch tensor ] flow values to target 1
        [ (N,) torch tensor ] flow values to target 2
        """
        srcx = ( coord_src_t,  src_feat_t,  batchsize )
        tar1 = ( coord_tar1_t, tar1_feat_t, batchsize )
        tar2 = ( coord_tar2_t, tar2_feat_t, batchsize )

        xsrc = self.source_inputlayer(srcx)
        xsrc = self.stem( xsrc )
        xsrc = self.resnet_layers( xsrc )
        xsrc = self.feature_layer( xsrc )

        xtar1 = self.target1_inputlayer(tar1)
        xtar1 = self.stem( xtar1 )
        xtar1 = self.resnet_layers( xtar1 )
        xtar1 = self.feature_layer( xtar1 )

        xtar2 = self.target1_inputlayer(tar2)
        xtar2 = self.stem( xtar2 )
        xtar2 = self.resnet_layers( xtar2 )
        xtar2 = self.feature_layer( xtar2 )

        xsrc  = self.source_outlayer(  xsrc )
        xtar1 = self.target1_outlayer( xtar1 )
        xtar2 = self.target2_outlayer( xtar2 )
        #print "source feature tensor: ",xsrc.shape,xsrc.grad_fn

        if npts1 is None:
            npts1 = self.neval
        if npts2 is None:
            npts2 = self.neval
        
        bstart_src  = 0
        bstart_tar1 = 0
        bstart_tar2 = 0
        if return_truth:
            truthvec1 = torch.zeros( (1,1,batchsize*self.neval), requires_grad=False, dtype=torch.int32 ).to( DEVICE )
            truthvec2 = torch.zeros( (1,1,batchsize*self.neval), requires_grad=False, dtype=torch.int32 ).to( DEVICE )
        
        for b in range(batchsize):
            if batchsize>1:
                nbatch_src = coord_src_t[:,2].eq(b).sum()
                nbatch_tar1 = coord_tar1_t[:,2].eq(b).sum()
                nbatch_tar2 = coord_tar2_t[:,2].eq(b).sum()
            else:
                nbatch_src  = coord_src_t.shape[0]
                nbatch_tar1 = coord_tar1_t.shape[0]
                nbatch_tar2 = coord_tar2_t.shape[0]
            
            bend_src = bstart_src + nbatch_src
            bend_tar1 = bstart_tar1 + nbatch_tar1
            bend_tar2 = bstart_tar2 + nbatch_tar2

            pred1,t1 = self.classify_sample( coord_src_t[bstart_src:bend_src,:],
                                             xsrc[bstart_src:bend_src,:],
                                             xtar1[bstart_tar1:bend_tar1,:],
                                             pair_flow1_v[b], DEVICE, return_truth,
                                             npts1 )
            pred2,t2 = self.classify_sample( coord_src_t[bstart_src:bend_src,:],
                                             xsrc[bstart_src:bend_src,:],
                                             xtar2[bstart_tar2:bend_tar2,:],
                                             pair_flow2_v[b], DEVICE, return_truth,
                                             npts2 )
            if return_truth:
                truthvec1[0,0,b*self.neval:b*self.neval+self.neval] = t1
                truthvec2[0,0,b*self.neval:b*self.neval+self.neval] = t2 
            bstart_src = bend_src
            bstart_tar1 = bend_tar1
            bstart_tar2 = bend_tar2
            
        #print "return results"
        if not return_truth:
            return pred1,pred2
        else:
            return pred1,pred2,truthvec1,truthvec2
                        
    def classify_sample(self,coord_src_t,feat_src_t,feat_tar_t,matchidx,DEVICE,return_truth,npts):

        from larcv import larcv
        larcv.load_pyutil()
        from larflow import larflow
        from ctypes import c_int
        
        coord_src_cpu = coord_src_t.to(torch.device("cpu"))
        
        # make random list of source indices
        nsamples = c_int()
        nsamples.value = self.neval
        nfilled = c_int()
        print "matchidx: ",matchidx.shape,matchidx.dtype            
        #print "make pairs of feature vectors to evaluate"
        
        # gather feature vector pairs
        matchidx = matchidx.to(DEVICE)
        srcfeats = torch.transpose( torch.index_select( feat_src_t, 0, matchidx[:npts,0] ), 0, 1 )
        tarfeats = torch.transpose( torch.index_select( feat_tar_t, 0, matchidx[:npts,1] ), 0, 1 )
        #print "feats: src=",srcfeats.grad_fn," tar=",tarfeats.grad_fn
        matchvec = torch.cat( (srcfeats,tarfeats), dim=0 ).reshape( (1,srcfeats.shape[0]+tarfeats.shape[0],srcfeats.shape[1]) )
        #print "match pair tensor made: ",matchvec.shape,matchvec.requires_grad,matchvec.grad_fn
        
        #print "nsrc indices used: ",nfilled,". run classifiers"

        pred = self.classifier(matchvec)
        if not return_truth:
            return pred,None
        else:
            return pred,matchidx[:,2]

                    
        
        
