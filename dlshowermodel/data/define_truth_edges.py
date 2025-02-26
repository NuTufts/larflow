import os,sys
import numpy as np

def make_true_edge_list( cluster_labels, instance_labels, particle_labels,
                         keypoint_labels, verbose=False, skip_cluster_ids=[-1,0] ):

    kptypes = [2,3,4] # shower, michel, delta

    if verbose:
        print("cluster labels: ",cluster_labels.shape)
        print("instance_labels: ",instance_labels.shape)
        print("particle_labels: ",particle_labels.shape)
        print("keypoint_labels: ",keypoint_labels.shape)

    clusterids = np.unique( cluster_labels )
    if verbose:
        print("num clusterids: ",len(clusterids))

    nkps_per_cluster = {}
    trackid_to_clusterid = {}
    clusterid_to_trackid = {}
    trackid_trunkcandidates = {}

    for cid in clusterids:

        if cid in skip_cluster_ids:
            continue
        
        # filter cluster points
        cluster_filter = (cluster_labels==cid)
        nclusterpts = cluster_filter.sum()
        if verbose:
            print("cluster id=",cid," npts=",nclusterpts)


        cluster_trackids = instance_labels[cluster_filter]
        trackids, tid_counts = np.unique(cluster_trackids, return_counts=True)
        if verbose:
            print("  trackids: ",trackids)
            print("  trackid counts: ",tid_counts)
        tid_counts = tid_counts[trackids!=0]
        trackids = trackids[ trackids!=0 ]
        max_trackid = -1
        max_tid_counts = -1
        if len(trackids)>0:
            idx_tid = np.argmax( tid_counts )
            max_trackid = trackids[idx_tid]
            max_tid_counts = tid_counts[idx_tid]
            # assign the truth trackid to this cluster based on most votes
            clusterid_to_trackid[cid] = max_trackid
        else:
            # this cluster is not associated to any true trajectory
            clusterid_to_trackid[cid] = 0
            continue

        # make a one-to-many map from trackid to cluster ids
        # this collects the clusters associated to a given trackid
        if max_trackid not in trackid_to_clusterid:
           trackid_to_clusterid[max_trackid] = [cid]
        else:
            trackid_to_clusterid[max_trackid].append(cid)

        # sum up the total number of pixels close to the keypoint
        showerkeypts = keypoint_labels[kptypes[0],cluster_filter]>0.5
        for kptype in kptypes[1:]:
            showerkeypts |= keypoint_labels[kptype,cluster_filter]>0.5
        nkps_per_cluster[cid] = showerkeypts.sum()
        showerkeypt_frac = float(showerkeypts.sum())/float(nclusterpts)

        if verbose:
            print("  npoints on keypoint: ",nkps_per_cluster[cid])
            print("  max trackid: ",max_trackid," counts=",max_tid_counts)

        # what is the partile type of this reco cluster
        # labels are
        # 0: no label
        # 1: electron
        # 2: photon
        # 3: muon
        # 4: proton
        # 5: pion/meson
        cluster_pid = particle_labels[cluster_filter]
        pids, pid_counts = np.unique(cluster_pid, return_counts=True)
        shower_counts = (cluster_pid==1).sum() + (cluster_pid==2).sum()
        shower_frac = float(shower_counts.sum())/float(nclusterpts)

        if verbose:
            print("  pids found: ",pids)
            print("  pid counts: ",pid_counts)
            print("  shower_counts: ",shower_counts," frac=",shower_frac)

        if shower_frac>0.5 and nkps_per_cluster[cid]>10.0:
            # label this a trunk cluster if it is majority true shower and has enough keypoints on it
            if max_trackid not in trackid_trunkcandidates:
                trackid_trunkcandidates[max_trackid] = cid
            else:
                # already had a trunk candidate for this trackid
                # replace if we have more keypoint pts on this cluster
                prev_cid = trackid_trunkcandidates[max_trackid]
                if nkps_per_cluster[cid]>nkps_per_cluster[prev_cid]:
                    trackid_trunkcandidates[max_trackid] = cid
    # end of loop over clusters

    if verbose:
        print(trackid_to_clusterid)
        print(trackid_trunkcandidates)    

    # now we can define edges
    edge_list = []
    for trackid in trackid_to_clusterid:
        cid_list = trackid_to_clusterid[trackid]
        if trackid not in trackid_trunkcandidates:
            continue
        trunk_cid = trackid_trunkcandidates[trackid]
        # make edge from cluster to trunk cluster
        # this includes a self-edge indicating a trunk cluster
        for cid in cid_list:
            edge_list.append( [cid,trunk_cid])
    
    edge_array = np.array(edge_list, dtype=np.int64)
    print("edge array")
    print(edge_array)
    return edge_array
        

        
            
if __name__ == "__main__":
    import dlshowermodel.data.larmatchhit_hdf5_reader as reader
    from larlite import larlite
    from ublarcvapp import ublarcvapp

    hdf_testfile = "../dataprep/test.h5"
    reader = reader.LArMatchHitHDF5Dataset( file_paths=[hdf_testfile],
                                            file_has_training_labels=True,
                                            file_has_larmatch_inputs=True,
                                            verbose=False )
    entry = reader[0]
    print(entry.keys())

    cluster_labels = entry['cluster_labels']
    point_pos = entry['shower_points']
    lms_filter = entry['lmshower_selection_mask'][0,:]

    print('lms_filter: ',lms_filter.shape)
    print('instance_labels.shape: ',entry['instanceids'].shape)
    print('keypoint_labels, pre-filter: ',entry['kpscores'].shape)
    instance_labels = entry['instanceids'][0,lms_filter[:]]
    particle_labels = entry['particleids'][0,lms_filter[:]]

    
    keypoint_labels = entry['kpscores'][:,lms_filter[:]]
    print('keypoint_labels, post-filter: ',keypoint_labels.shape)

    llfile = "/home/twongjirad/working/data/mcc9_v40a_dl_run3b_NC_pi0_overlay_CV/merged_dlreco_mcc9_v40a_dl_run3b_NC_pi0_overlay_CV_aa444faa-530a-4fd7-b43f-b501bc221880.root"
    io = larlite.storage_manager(larlite.storage_manager.kREAD)
    io.add_in_filename( llfile )
    io.open()
    io.go_to(0)

    mcpg = ublarcvapp.mctools.MCPixelPGraph()
    mcpg.buildgraphonly( io )
    mcpg.printGraph(0,0)

    make_true_edge_list( cluster_labels, instance_labels, particle_labels, keypoint_labels )
    
