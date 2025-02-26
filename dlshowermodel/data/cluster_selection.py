import numpy as np

def cluster_filter_by_size( cluster_labels, min_npoints=30, skip_cluster_ids=[-1] ):

    cid_list = np.unique(cluster_labels)
    cid_remap = {}
    xcid = 0
    for cid in cid_list:
        if cid in skip_cluster_ids:
            continue
        cluster_filter = cluster_labels==cid
        if cluster_filter.sum()<min_npoints:
            cluster_labels[cluster_filter] = -1 # re-label points in this cluster as noise point
        else:
            cid_remap[cid] = xcid
            xcid += 1
    
    cid_remap_np = np.zeros( len(cid_remap), dtype=np.long )
    for cid,xcid in cid_remap.items():
        cid_remap_np[xcid] = cid

    return cid_remap_np