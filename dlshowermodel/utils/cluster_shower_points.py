import torch
from .dbscan_torch import dbscan_torch
from .densityawaresampling import DensityAwareSemanticSampling

def cluster_lmshower_points( pos, lm_logits, shower_prob, 
                            dbscan_eps=0.5, dbscan_minsamples=5,
                            lmscore_threshold=0.20, ssnet_threshold=0.5,
                            lmtruept_index=-1,
                            use_scikit=False ):
    """
    """

    N = shower_prob.shape[0]

    if lm_logits.shape[0]!=N:
        raise ValueError("number of points in lm and ssnet tensor does not match")
    if pos.shape[0]!=N:
        raise ValueError("number of points in pos tensor does not match")

    if len(lm_logits.shape)>1:
        lm = lm_logits[:,lmtruept_index].squeeze() # shape (N)
    else:
        lm = lm_logits

    # filter out points with high ssnet and lm confidence
    lmsfilter = (lm > lmscore_threshold)*(shower_prob > ssnet_threshold)

    lms_pos = pos[lmsfilter[:],:]

    print("Run DBSCAN on shower-larmatch points")
    if not use_scikit and lms_pos.is_cuda:
        # try the torch dbscan ...
        # note: noise points are labeled with -1
        labels = dbscan_torch( lms_pos, dbscan_eps, dbscan_minsamples )
    else:
        from sklearn.cluster import DBSCAN

        # go back to numpy
        np_lms_pos = lms_pos.detach().cpu().numpy()

        clustering = DBSCAN( eps=0.5, min_samples=dbscan_minsamples ).fit( np_lms_pos )
        labels = torch.from_numpy( clustering.labels_ ).to( lms_pos.device )

    return lms_pos, labels, lmsfilter

    

class ClusterShowerPoints:
    def __init__(self, max_samples_per_cluster=16, 
                dbscan_eps=0.5, dbscan_minsamples=5,
                min_cluster_size=60,
                lmscore_threshold=0.5, lmtruept_index=-1 ):

        self.max_samples_per_cluster = max_samples_per_cluster
        self.dbscan_eps = dbscan_eps
        self.dbscan_minsamples = dbscan_minsamples
        self.lmscore_threshold = lmscore_threshold
        self.lmtruept_index = lmtruept_index
        self.min_cluster_size = min_cluster_size
        self.max_samples_per_cluster = max_samples_per_cluster
        self.dass_alg = DensityAwareSemanticSampling(n_samples=max_samples_per_cluster)

    def process_event_points(self, pos, features, lm_logits, ssnet_logits, use_scikit=True ):
        """
        """
        Np, Nf = features.shape 
        print("pos.shape: ",pos.shape)
        print("lm_logits.shape: ",lm_logits.shape)
        print("ssnet_logits.shape: ",ssnet_logits.shape)

        lms_pos, cluster_labels, lms_filter = cluster_lmshower_points( pos, lm_logits, ssnet_logits,
                                            dbscan_eps=self.dbscan_eps, dbscan_minsamples=self.dbscan_minsamples,
                                            lmscore_threshold=self.lmscore_threshold, 
                                            lmtruept_index=self.lmtruept_index,
                                            use_scikit=use_scikit  )
        print("lms_pos.shape: ",lms_pos.shape)
        print("cluster_labels.shape: ",cluster_labels.shape)
        print("max cluster id: ",torch.unique(cluster_labels).max())
        print("lms_filter.shape: ",lms_filter.shape)


        # filter down features
        lms_features = features[lms_filter[:],:]

        # loop over clusters and subsample representative feature vectors
        cluster_sampled_pos_v = []
        cluster_sampled_feat_v = []
        clusterids = torch.unique( cluster_labels )
        max_cid = clusterids.max()

        cid_remap = {}
        icid = 0
        for cid in range(max_cid):
            if cid<0:
                continue
            cid_filter = cluster_labels==cid
            cpos   = lms_pos[ cid_filter, : ]

            if cpos.shape[0]<self.min_cluster_size:
                # return the labels for this cluster to the noise label (-1)
                cluster_labels[cid_filter] = -1
                continue

            cid_remap[cid] = icid
            icid += 1

            cfeats = lms_features[ cluster_labels==cid, : ]

            csampled_feats = torch.zeros( (self.max_samples_per_cluster,Nf)).to(pos.device)
            csampled_pos   = torch.zeros( (self.max_samples_per_cluster,cpos.shape[-1])).to(pos.device)

            if cpos.shape[0]>0:
                #print("Run DASS on cluster of size=",cpos.shape)

                if cpos.shape[0]<5000:
                    dass_results = self.dass_alg( cpos, cfeats )
                    nsamples = dass_results["sampled_points"].shape[0]

                    if nsamples<self.max_samples_per_cluster:
                        csampled_feats[:nsamples] = dass_results["sampled_features"]
                        csampled_pos[:nsamples]   = dass_results["sampled_points"]
                    else:
                        csampled_feats = dass_results["sampled_features"]
                        csampled_pos = dass_results["sampled_points"]
                else:
                    # for big showers, void the big calculations
                    # we find the first pca, sample points along the line
                    from sklearn.decomposition import PCA
                    import numpy as np
                    print("Large shower subsampling: npts=",cpos.shape[0])
                    np_cpoints = cpos.detach().cpu().numpy()
                    np_cfeats  = cfeats.detach().cpu().numpy()
                    pca = PCA(n_components=3)
                    pca.fit(np_cpoints)
                    pos_pca = pca.transform(np_cpoints)

                    # restrict points to the core
                    dist2pca1 = np.sum( pos_pca[:,1:]*pos_pca[:,1:], axis=1 )
                    corepoints = dist2pca1<4.0 # 2 cm on the core
                    core_pca   = pos_pca[corepoints]
                    core_pos   = np_cpoints[corepoints]
                    core_feats = np_cfeats[corepoints]

                    core_x = np.expand_dims(core_pca[:,0],1) # (M,1)
                    pos_pca1 = np.linspace(np.min(pos_pca[:,0]),np.max(pos_pca[:,0]), self.max_samples_per_cluster) # (Ns)
                    core_dx = core_x-pos_pca1
                    core_dx = core_dx*core_dx
                    #print("core_dx: ",core_dx.shape)
                    sample_indices = np.squeeze( np.argmin( core_dx, axis=0 ) )
                    #print("sample_indices: ",sample_indices.shape)
                    #print(sample_indices)
                    csampled_pos   = torch.from_numpy( np.take( core_pos, sample_indices, axis=0 ) ).to(pos.device)
                    csampled_feats = torch.from_numpy( np.take( core_feats, sample_indices, axis=0 ) ).to(pos.device)
                    #print("csampled_pos: ",csampled_pos.shape)


            cluster_sampled_pos_v.append( csampled_pos.unsqueeze(0) )
            cluster_sampled_feat_v.append( csampled_feats.unsqueeze(0) )

        cluster_sampled_pos_t = torch.cat(cluster_sampled_pos_v, 0)
        cluster_sampled_feat_t = torch.cat(cluster_sampled_feat_v, 0)

        # relabel the index of passing clusters
        for cid in range(max_cid):
            if cid in cid_remap:
                cluster_labels[ cluster_labels==cid ] = cid_remap[cid]
        print("number of clusters: ",len(cid_remap))

        return {"shower_points":lms_pos,
                "shower_feats":lms_features,
                "cluster_labels":cluster_labels,
                "cluster_sampled_pos":cluster_sampled_pos_t,
                "cluster_sampled_feat":cluster_sampled_feat_t,
                "lmshower_selection_mask":lms_filter}

    def make_shower_keypoint_cluster_labels(self, true_keypoint, shower_points, cluster_labels, lms_filter ):
        """
        """
        # combine the keypoint label scores for all shower keypoint types
        print("keypoint labels: ",true_keypoint.shape)
        allshower_kp_labels = torch.max( true_keypoint[2:,:], dim=1 )
        # filter the spacepoints based on the larmatch+shower score
        allshower_kp_labels = allshower_kp_labels[lms_filter[:]]



    def make_truth_assignments(self, instanceids, origin, ):
        pass



    
        
