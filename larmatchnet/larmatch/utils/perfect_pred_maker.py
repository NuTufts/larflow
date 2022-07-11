import os,sys
import torch


def make_perfect_larmatch( lmtruth, DEVICE ):
    perfect_lm = torch.zeros( (lmtruth.shape[0],2,lmtruth.shape[-1]) ).to(DEVICE)
    print("perfect_lm: ",perfect_lm.shape)

    true_lm  = lmtruth.ge(0.5)
    false_lm = lmtruth.lt(0.5)

    # soft label solution
    p = torch.clamp(lmtruth, 0.0001,0.99)
    #print("p: ",p.shape)
    #print(p[0,0,0:10])

    x = 0.5*(torch.log(p)-torch.log(1-p))
    #print("x: ",x.shape)
    #print(x[0,0,0:10])

    perfect_lm[:,1,:] = x[:,0,:]
    perfect_lm[:,0,:] = -x[:,0,:]

    sm_perfect = torch.softmax(perfect_lm, dim=1)
    #print("sm_perfect: ",sm_perfect.shape)
    #print(sm_perfect[0,1,:10])
    sm_diff = torch.abs(sm_perfect[:,1,:]-p[:,0,:]).sum()
    print("softmax diff check (should be close to zero): ",sm_diff)
    return perfect_lm
    
        
def make_perfect_ssnet( ssnet_truth, DEVICE, logitval=2.0 ):
    pred = torch.zeros( (ssnet_truth.shape[0], 7, ssnet_truth.shape[2]), dtype=torch.float ).to(DEVICE)
    for c in range(0,7):
        c_mask = ssnet_truth[:,0,:]==c
        c_neg_mask = ssnet_truth[:,0,:]!=c
        pred[:,c,:][c_mask]     = logitval
        pred[:,c,:][c_neg_mask] = -logitval
    return pred
