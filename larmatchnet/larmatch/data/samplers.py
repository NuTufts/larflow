import os,sys
import numpy as np

"""
Functions that resamples the larmatch data
"""

def larmatch_example_balancer( data, max_nspacepoints_returned=300000,
    exclude_ghosts=True,
    ignore_array_list=['coord_0', 'feat_0', 
    'coord_1', 'feat_1', 
    'coord_2', 'feat_2',
    'keypoint_truth_pos','keypoint_truth_kptype_pdg_trackid']  ):
    """
    We resample the batch to balance the batch of positive and negative examples
    to 50% positive and 50% negative.
    """

    # first count the number of positive and negative examples
    lmtruth = data['matchtriplet_v'][:,3] # row==1 if true energy deposit
    truemask  = lmtruth==1
    falsemask = lmtruth==0

    npos = truemask.sum()
    nneg = falsemask.sum()
    
    nsampletrue = int(max_nspacepoints_returned/2)
    passallpos = False
    if nsampletrue > npos:
        nsampletrue = npos
        passallpos = True

    # sample for false examples
    nsampleneg = int(max_nspacepoints_returned/2)
    passallneg = False
    if nsampleneg > nneg:
        passallneg = True
        nsampleneg = nneg

    #print("npos: ",npos)
    #print("nneg: ",nneg)
    #print("ntotal: ",lmtruth.shape[0])

    # make a master mask
    combinedmask = np.zeros( lmtruth.shape[0], dtype=np.int32 )
    #print("premask: ",combinedmask.sum())

    # sub-sample positive examples
    pos_frac = (float(nsampletrue)/float(npos))
    #print("pos_frac: ",pos_frac)
    if not passallpos:
        combinedmask[truemask[:]] = np.random.random(npos)<pos_frac
    else:
        combinedmask[truemask[:]] = 1
    # sub-sample neg examples
    neg_frac = (float(nsampleneg)/float(nneg))
    #print("neg frac: ",neg_frac)
    if not passallneg:
        combinedmask[falsemask[:]] = np.random.random(nneg)<neg_frac
    else:
        combinedmask[falsemask[:]] = 1

    #print("combinedmask.sum()=",combinedmask.sum())
    
    # subsample data
    sampled_data = {}
    for name in data:
        #print(name," ",data[name].shape)
        if name in ignore_array_list:
            sampled_data[name] = data[name]
        else:
            try:
                if name in ['keypoint_truth','keypoint_weight']:
                    sampled_data[name] = data[name][:,combinedmask[:]==1]
                elif name in ['paf_label']:
                    sampled_data[name] = data[name][:,:,combinedmask[:]==1]
                elif name in ['spacepoints']:
                    sampled_data[name] = data[name][combinedmask[:]==1,:]
                else:
                    sampled_data[name] = data[name][combinedmask==1]
                #print("  sample ",name," ",data[name].shape," to ",sampled_data[name].shape)
            except:
                raise ValueError("Cannot sample array name=",name," with shape=",data[name].shape)

    sampled_data['larmatch_weight'] = make_lm_weights( sampled_data )
    sampled_data['keypoint_weight'] = make_kp_weights( sampled_data, exclude_ghosts=exclude_ghosts )
    sampled_data['ssnet_weight']    = make_ssnet_weights( sampled_data, exclude_ghosts=exclude_ghosts )
    sampled_data['paf_weight']      = make_paf_weights( sampled_data, exclude_ghosts=exclude_ghosts )

    return sampled_data

def make_lm_weights( entrydata ):
    lmtruth = entrydata['matchtriplet_v'][:,3]
    truemask  = lmtruth==1
    falsemask = lmtruth==0
    npos = float(truemask.sum())
    nneg = float(falsemask.sum())
    lmweight = np.zeros( (lmtruth.shape[0]), dtype=np.float32 )
    if npos==0 or nneg==0:
        lmweight[:,:] = 1.0/(npos+nneg)
    else:
        lmweight[truemask[:]]  = 0.5/npos
        lmweight[falsemask[:]] = 0.5/nneg
    return lmweight

def make_kp_weights( entrydata, exclude_ghosts=True ):
    lmtruth = entrydata['matchtriplet_v'][:,3]
    lmpos = lmtruth==1
    lmneg = lmtruth==0
    kptruescore = entrydata['keypoint_truth']
    posmask = (entrydata['keypoint_truth']>1.0e-3)
    negmask = (entrydata['keypoint_truth']<1.0e-3)
        
    nkpclasses = kptruescore.shape[0]
    nexamples  = kptruescore.shape[1]
    kpweight = np.zeros( (nkpclasses,nexamples), dtype=np.float32 )
    #print('kpweight.shape: ', kpweight.shape)
    for iclass in range(nkpclasses):
        posclass = posmask[iclass,:]
        negclass = negmask[iclass,:]
        if exclude_ghosts:
            posclass[lmneg[:]] = False
            negclass[lmneg[:]] = False
        npos = float(posclass.sum())
        nneg = float(negclass.sum())
        #print("kpclass[",iclass,']: npos=',npos," nneg=",nneg," nghost=",lmneg.sum()," npoints=",nexamples," checksum=",npos+nneg+lmneg.sum())
        nnorm = 0.5
        if npos==0 or nneg==0:
            nnorm = 1.0
        if npos>0:
            kpweight[ iclass, posclass[:] ] = nnorm/npos
        if nneg>0:
            kpweight[ iclass, negclass[:] ] = nnorm/nneg
        kpweight[ iclass, lmneg[:] ] = 0.0
    return kpweight

def make_ssnet_weights( entrydata, exclude_ghosts=True ):
    lmtruth = entrydata['matchtriplet_v'][:,3]
    lmpos = lmtruth==1
    lmneg = lmtruth==0
    ssnettruth = entrydata['ssnet_truth']
    #print('ssnet_truth.shape: ',ssnettruth.shape)
    #print('ssnet class labels: ',np.unique( ssnettruth ))
    nclasses  = 5
    #print("ssnet nclasses: ",nclasses)
    nexamples = lmtruth.shape[0]
    weights = np.zeros( (nexamples), dtype=np.float32 )
    nclass = {}
    cmask_v = {}
    nnorm = 0.0
    for iclass in range(0,nclasses):
        cmask = ssnettruth==iclass
        if exclude_ghosts:
            cmask[lmneg[:]] = False
        nclass[iclass] = cmask.sum()
        if nclass[iclass]>0.0:
            nnorm += 1.0
        cmask_v[iclass] = cmask
    if nnorm>0.0:
        nnorm = 1.0/nnorm
    for iclass in range(0,nclasses):
        if nclass[iclass]>0.0:
            w = nnorm/float(nclass[iclass])
            weights[ cmask_v[iclass] ] = w
    # blank out BG and blank out ghosts
    bgmask = ssnettruth==-1 # ghost mask
    weights[ bgmask ] = 0.0
    if exclude_ghosts:
        weights[ lmneg ] = 0.0
    return weights
    


def make_paf_weights( entrydata, exclude_ghosts=True ):
    lmtruth = entrydata['matchtriplet_v'][:,3]
    lmpos = lmtruth==1
    lmneg = lmtruth==0
    paflabel = entrydata['paf_label'].squeeze()
    #print('paflabel.shape=',paflabel.shape)
    nexamples = lmtruth.shape[0]
    weights = np.zeros( (nexamples), dtype=np.float32 )
    labelsum = np.sum( paflabel*paflabel, axis=0 )
    maskzero = labelsum < 0.1
    maskvec  = labelsum > 0.1
    nzero = maskzero.sum()
    nnonzero = maskvec.sum()
    #print("paf nzero=",nzero)
    if nnonzero>0:
        weights[maskvec] = 1.0/float(nnonzero)
    if exclude_ghosts:
        weights[lmneg] = 0.0
    return weights
    

    
