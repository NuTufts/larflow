import torch
from .dbscan_torch import dbscan_torch
from .densityawaresampling import DensityAwareSemanticSampling

def cluster_lmshower_points( pos, lm_logits, ssnet_logits, 
                            dbscan_eps=0.5, dbscan_minsamples=100,
                            lmscore_threshold=0.5, lmtruept_index=-1,
                            use_scikit=False ):
    """
    """
    if len(ssnet_logits.shape)!=2:
        raise ValueError("expected dim for ssnet_logits tensor is not 2. Expect (C,N) tensor.")

    N,C = ssnet_logits.shape

    if lm_logits.shape[0]!=N:
        raise ValueError("number of points in lm and ssnet tensor does not match")
    if pos.shape[0]!=N:
        raise ValueError("number of points in pos tensor does not match")

    ssnet_probs = torch.softmax( ssnet_logits, 1) # normalize along dim-1 (length C), out shape (N,)
    shower_prob = torch.sum( ssnet_probs[:,:2], dim=1 ) # electron + photon scores

    if len(lm_logits.shape)>1:
        lm = lm_logits[:,lmtruept_index].squeeze() # shape (N)
    else:
        lm = lm_logits
    
    lmscore = shower_prob*lm

    # filter out points with high ssnet and lm confidence
    lmsfilter = lmscore > lmscore_threshold

    lms_pos = pos[lmsfilter[:],:]

    if not use_scikit:
        # try the torch dbscan ...
        # note: noise points are labeled with -1
        labels = dbscan_torch( lms_pos, dbscan_eps, dbscan_minsamples )
    else:
        import sklearn
        from sklearn.cluster import DBSCAN

        # go back to numpy
        np_lms_pos = lms_pos.detach().cpu().numpy()

        clustering = DBSCAN( eps=0.5, min_samples=dbscan_minsamples ).fit( np_lms_pos )
        labels = torch.from_numpy( clustering.labels_ ).to( pos.device )

    return lms_pos, labels, lmsfilter

    

class ClusterShowerPoints:
    def __init__(self, max_samples_per_cluster=16, 
                dbscan_eps=0.5, dbscan_minsamples=100,
                lmscore_threshold=0.5, lmtruept_index=-1 ):

        self.max_samples_per_cluster = max_samples_per_cluster
        self.dbscan_eps = dbscan_eps
        self.dbscan_minsamples = dbscan_minsamples
        self.lmscore_threshold = lmscore_threshold
        self.lmtruept_index = lmtruept_index
        self.max_samples_per_cluster = max_samples_per_cluster
        self.dass_alg = DensityAwareSemanticSampling(n_samples=max_samples_per_cluster)

    def process_event_points(self, pos, features, lm_logits, ssnet_logits ):
        """
        """
        Np, Nf = features.shape 
        print("pos.shape: ",pos.shape)
        print("lm_logits.shape: ",lm_logits.shape)
        print("ssnet_logits.shape: ",ssnet_logits.shape)

        lms_pos, cluster_labels, lms_filter = cluster_lmshower_points( pos, lm_logits, ssnet_logits,
                                            dbscan_eps=self.dbscan_eps, dbscan_minsamples=self.dbscan_eps,
                                            lmscore_threshold=self.lmscore_threshold, 
                                            lmtruept_index=self.lmtruept_index,
                                            use_scikit=False  )
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

        for cid in clusterids:
            if cid<0:
                continue
            cpos   = lms_pos[ cluster_labels==cid, : ]
            cfeats = lms_features[ cluster_labels==cid, : ]

            dass_results = self.dass_alg( cpos, cfeats )
            nsamples = dass_results["sampled_points"].shape[0]

            if nsamples<self.max_samples_per_cluster:
                csampled_feats = torch.zeros( (self.max_samples_per_cluster,Nf)).to(pos.device)
                csampled_pos   = torch.zeros( (self.max_samples_per_cluster,cpos.shape[-1])).to(pos.device)
                csampled_feats[:nsamples] = dass_results["sampled_features"]
                csampled_pos[:nsamples]   = dass_results["sampled_points"]
            else:
                csampled_feats = dass_results["sampled_features"]
                csampled_pos = dass_results["sampled_points"]

            cluster_sampled_pos_v.append( csampled_pos.unsqueeze(0) )
            cluster_sampled_feat_v.append( csampled_feats.unsqueeze(0) )

        cluster_sampled_pos_t = torch.cat(cluster_sampled_pos_v, 0)
        cluster_sampled_feat_t = torch.cat(cluster_sampled_feat_v, 0)

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



    
        
