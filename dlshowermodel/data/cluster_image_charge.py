import numpy as np
from larflow import larflow

def get_cluster_image_pixels( cluster_labels, matchtriplets, wireimage_list, larcv_image2d_list,
                            threshold=10.0, drow=2, dcol=2, skip_cluster_indices=[-1] ):
    """
    cluster_labels: (N,) np.long array containing cluster indices
    matchtriplets: (N,3) np.long array with each row giving the index of the pixel in the wireimage 2D array
    wiremage_list: list of (N,3) np.float array. Each array contains above threshold wire plane image pixels
                    with each row being (row,col,pixelvalue)
    larcv_image2d_list: list of larcv::Image2D objects containing the full image data for each wire plane
                    we use this to sum pixels around the pixels that the cluster spacepoints project into
    threshold: float When summing the pixels in the image2d, only include pixels above the threshold
    drow: int Include neighboring 'drow' rows above and below the cluster pixel
    dcol: int Include neighboring 'dcol' cols left and right next to the cluster pixels
    """
    clusterimagemasker = larflow.reco.ClusterImageMask()
    clusterids = np.unique( cluster_labels )
    max_cid = int(np.max(clusterids))
    nplanes = len(wireimage_list)
    cluster_pixelsum_v = []
    cluster_pixelmask = {}
    cluster_pixelmaskq = {}
    npixels = 0
    for cid in clusterids:
        if cid in skip_cluster_indices:
            continue
        cluster_filter = cluster_labels==cid
        cluster_triplets = matchtriplets[cluster_filter[:],:]
        # get row,col array
        cluster_planemasks = []
        cluster_planemaskq = []
        cid_pixsum = np.zeros((1,3))
        for p,img in enumerate(wireimage_list):
            cluster_pixels = img[ cluster_triplets[:,p],:2].astype( np.long )
            maskresults = clusterimagemasker.getClusterImageChargeSum( cluster_pixels, larcv_image2d_list[p], threshold, drow, dcol)
            cid_pixsum[0,p] = maskresults['pixelsum']
            npixels += maskresults['pixelmask'].shape[0]
            cluster_planemasks.append( maskresults['pixelmask'])
            cluster_planemaskq.append( maskresults['pixelvalues'])
        cluster_pixelmask[cid]  = cluster_planemasks
        cluster_pixelmaskq[cid] = cluster_planemaskq
        cluster_pixelsum_v.append( cid_pixsum )
    cluster_pixelsum = np.concatenate( cluster_pixelsum_v, axis=0 )
    #print("cluster plane pixelsums")
    #print(cluster_pixelsum)

    # put pixel list into one big array with colums [ row, col, clusterid, planeid ]
    pixel_list = np.zeros( (npixels, 4), dtype=np.int32)
    pixval_list = np.zeros( (npixels,1) )
    ipix = 0
    for cid in clusterids:
        if cid not in cluster_pixelmask:
            continue
        for p,pixmask in enumerate( cluster_pixelmask[cid] ):
            pixel_list[ipix:ipix+pixmask.shape[0],:2] = pixmask
            pixel_list[ipix:ipix+pixmask.shape[0],2] = cid
            pixel_list[ipix:ipix+pixmask.shape[0],3] = p
            pixval_list[ipix:ipix+pixmask.shape[0],0] = cluster_pixelmaskq[cid][p]
            ipix += pixmask.shape[0]

    return {'cluster_pixelsum':cluster_pixelsum,
            'cluster_pixelmask':pixel_list,
            'cluster_pixelmaskq':pixval_list}

if __name__ == "__main__":

    import torch
    from larcv import larcv
    from larlite import larlite


    from dlshowermodel.data.larmatchhit_hdf5_writer import LArMatchHitHDF5Writer
    from .cluster_selection import cluster_filter_by_size

    lmwriter = LArMatchHitHDF5Writer()
    num_max_spacepoints = 10000000
    process_truth_labels = True
    triplet_key = 'matchtriplet'

    # testing
    dlmerged = "/home/twongjirad/working/data/mcc9_v40a_dl_run3b_NC_pi0_overlay_CV/merged_dlreco_mcc9_v40a_dl_run3b_NC_pi0_overlay_CV_aa444faa-530a-4fd7-b43f-b501bc221880.root"

    iolcv = larcv.IOManager( larcv.IOManager.kREAD, "larcv", larcv.IOManager.kTickBackward )
    iolcv.add_in_file( dlmerged )
    iolcv.reverse_all_products()
    iolcv.initialize()

    ioll = larlite.storage_manager( larlite.storage_manager.kREAD )
    ioll.add_in_filename( dlmerged )
    ioll.open()

    iolcv.read_entry(0)
    ioll.go_to(0)

    # convert the data and store into self.entry_data
    lmwriter.larlite_larcv_to_hdf5_entry( ioll, iolcv, process_truth_labels, num_max_spacepoints )
    entrydata = lmwriter.entry_data.pop()
    print("processed larcv/larlite info dict keys: ",entrydata.keys())

    # make a trial cluster
    instanceids = entrydata['instanceid_label']
    matchtriplets = entrydata['matchtriplet'][:,:3]

    cluster_filter = instanceids==42
    print("trackid=42: npts=",cluster_filter.sum())
    cluster_triplets = torch.from_numpy( matchtriplets[cluster_filter[:],:] )

    img0 = torch.from_numpy( entrydata['wireimage_plane0'] )
    img1 = torch.from_numpy( entrydata['wireimage_plane1'] )
    img2 = torch.from_numpy( entrydata['wireimage_plane2'] )
    print("wireimage_plane0: ",img0.shape)
    print("wireimage_plane1: ",img1.shape)
    print("wireimage_plane2: ",img2.shape)

    



