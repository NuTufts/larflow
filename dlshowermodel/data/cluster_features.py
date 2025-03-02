import os,sys
import numpy as np

def get_centroids(shower_points, cluster_labels, skip_cluster_ids=[-1], cid_list=None ):
    #print("shower_points: ",shower_points.shape)
    if cid_list is None:
        cid_list = np.unique(cluster_labels)
    cluster_centroids = []
    for cid in cid_list:
        # skip cluster ids (default -1 or 0 are used to tag noise cluster)
        if cid in skip_cluster_ids:
            continue
        cid_filter = cluster_labels==cid
        if cid_filter.sum()==0:
            continue
        pos = shower_points[cid_filter,:]
        mean_pos = np.mean( pos, axis=0 )
        #print("mean_pos: ",mean_pos.shape)
        cluster_centroids.append( np.expand_dims(mean_pos, 0) )
    out = np.concatenate( cluster_centroids, axis=0 )
    return out

def get_pc_axes( shower_points, cluster_labels, skip_cluster_ids=[-1], cid_list=None, verbose=False):
    from sklearn.decomposition import PCA

    if verbose:
        print("cluster_features.py:get_pc_axes")
        print("  [in] shower_points: ",shower_points.shape)
        print("  [in] cluster_labels: ",cluster_labels.shape)

    if cid_list is None:
        cid_list = np.unique(cluster_labels)

    cluster_feat_v = []

    for cid in cid_list:
        if cid in skip_cluster_ids:
            continue
        cid_filter = cluster_labels==cid
        if cid_filter.sum()==0:
            continue
        if verbose:
            print("cluster id[",cid,"] --------------") 
            print(" npoints=",cid_filter.sum())

        pos = shower_points[cid_filter,:]
        pca = PCA(n_components=3)
        pca.fit(pos)
        pos_pca = pca.transform(pos)
        pca_bounds = np.zeros(9) # (xmin, xmax, ...., xlen, ylen, zlen)
        for v in range(3):
            pca_bounds[2*v+0] = np.min(pos_pca[:,v])
            pca_bounds[2*v+1] = np.max(pos_pca[:,v])
            pca_bounds[6+v]   = np.abs( pca_bounds[2*v+1]-pca_bounds[2*v+0] )

        
        if verbose:
            print("pca components: ")
            print(pca.components_)
            print("explained variance:")
            print(pca.explained_variance_)

        cluster_feat = [  pca.components_[0,:], pca.components_[1,:], pca.components_[2,:], 
                            pca.explained_variance_, pca_bounds ]
        cluster_feat = np.concatenate( cluster_feat )
        cluster_feat_v.append( np.expand_dims( cluster_feat, 0 ) )
        if verbose:
            print("feat vector: ",cluster_feat)

    out = np.concatenate( cluster_feat_v, axis=0 )
    if verbose:
        print("outshape: ",out.shape)
        print(out[-3:,:])
    return out


if __name__== "__main__":

    # example of running some of the cluster feature code
    import dlshowermodel.data.larmatchhit_hdf5_reader as reader
    from .cluster_selection import cluster_filter_by_size

    hdf_testfile = "../test.h5"
    reader = reader.LArMatchHitHDF5Dataset( file_paths=[hdf_testfile],
                                            file_has_training_labels=True,
                                            file_has_larmatch_inputs=True,
                                            file_has_mctruth_labels=True )
    entry = reader[0]
    print(entry.keys())

    cluster_labels = entry['cluster_labels']
    cid_remap = cluster_filter_by_size(cluster_labels[0], min_npoints=30 )
    print("cluster_labels: ",cluster_labels.shape)
    print("number of filtered clusters: ",len(cid_remap))

    point_pos = entry['shower_points']
    lms_filter = entry['lmshower_selection_mask'][0,:]

    centroids = get_centroids( point_pos, cluster_labels[0])
    print("centroid feats: ",centroids.shape)

    pca_feats = get_pc_axes( point_pos, cluster_labels[0] )
    print("pca_feats: ",pca_feats.shape)