import os,sys
import numpy as np

# def match_showerkeypoints_to_clusters( points, cluster_labels, track_to_clusterids, trackid_keypt_pos ):
#     pass

def make_true_edge_list( cluster_labels, points, instance_labels, particle_labels,
                         keypoint_labels, keypoint_data, verbose=False, debug=False, skip_cluster_ids=[-1,0] ):

    if verbose:
        print("cluster labels: ",cluster_labels.shape)
        print("points: ",points.shape)
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
    trackid_keypt_pos = {}
    trackid_trunk_mindist2keypoint = {}

    kpids = keypoint_data[:,-1].astype(np.int64)
    for kpdata in keypoint_data:
        trackid = int(kpdata[-1])
        kppos = kpdata[4:7]
        trackid_keypt_pos[trackid] = kppos
        if debug:
            print("true keypoint: trackid=",trackid," pos=",kppos)

    for cid in clusterids:

        if cid in skip_cluster_ids:
            continue
        
        # filter cluster points
        cluster_filter = (cluster_labels==cid)
        nclusterpts = cluster_filter.sum()
        if debug:
            print("cluster id=",cid," npts=",nclusterpts)

        cluster_trackids = instance_labels[cluster_filter]
        trackids, tid_counts = np.unique(cluster_trackids, return_counts=True)
        if debug:
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
        if debug:
            print("  max trackid: ",max_trackid," counts=",max_tid_counts)

        # make a one-to-many map from trackid to cluster ids
        # this collects the clusters associated to a given trackid
        if max_trackid not in trackid_to_clusterid:
           trackid_to_clusterid[max_trackid] = [cid]
        else:
            trackid_to_clusterid[max_trackid].append(cid)

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
        if debug:
            print("  pids found: ",pids)
            print("  pid counts: ",pid_counts)
            print("  shower_counts: ",shower_counts," frac=",shower_frac)

        # determine trunk cluster
        # [option 1] sum up the total number of pixels close to the keypoint
        # showerkeypts = np.sum(keypoint_labels[3:,cluster_filter],axis=0)
        # if debug:
        #     print("  showerkeypts: ",showerkeypts.shape," min=",np.min(showerkeypts)," max=",np.max(showerkeypts))
        # nkps_per_cluster[cid] = (showerkeypts>0.1).sum()
        # showerkeypt_frac = float(nkps_per_cluster[cid])/float(nclusterpts)
        # if debug:
        #     print("  npoints on keypoint: ",nkps_per_cluster[cid],"  frac=",showerkeypt_frac)
        
        # [option 2] match to true keypoint positions, choose closest cluster
        if max_trackid in trackid_keypt_pos:
            cluster_pos = points[ cluster_filter, :]
            kppos = trackid_keypt_pos[max_trackid]
            dist2kp = cluster_pos - np.expand_dims(kppos,0) # (N,3) - (1,3): should repeat (1,3) to make (N,3)
            dist2kp = np.min(np.sum( dist2kp*dist2kp, axis= 1 )) # min((N))
            if debug:
                print("  distance to keypoint with same id: ",dist2kp)
            if max_trackid not in trackid_trunk_mindist2keypoint:
                trackid_trunk_mindist2keypoint[max_trackid] = dist2kp
                trackid_trunkcandidates[max_trackid] = cid
            elif dist2kp<trackid_trunk_mindist2keypoint[max_trackid]:
                trackid_trunk_mindist2keypoint[max_trackid] = dist2kp
                trackid_trunkcandidates[max_trackid] = cid


    # end of loop over clusters

    # check if some trackids do not have a trunk candidate
    for trackid in trackid_to_clusterid:
        if trackid not in trackid_trunkcandidates:
            # check if its close to one of the keypt ids
            diffid = np.abs( kpids - trackid )
            isclose = kpids[ diffid==1 ] # merges pairproduction? # a nice calibration sample kind of
            for closeid in isclose:
                # transfer cid list to trackid with trunk
                if closeid in trackid_to_clusterid and closeid!=trackid:
                    trackid_to_clusterid[closeid] += trackid_to_clusterid[trackid]
                    break

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
    if verbose:
        print("edge array")
        print(edge_array)
    return edge_array
        

        
            
if __name__ == "__main__":
    import dlshowermodel.data.larmatchhit_hdf5_reader as reader
    from larlite import larlite
    from ublarcvapp import ublarcvapp

    hdf_testfile = "../dataprep/test_bnbnue_corsika_e1.h5"
    reader = reader.LArMatchHitHDF5Dataset( file_paths=[hdf_testfile],
                                            file_has_training_labels=True,
                                            file_has_larmatch_inputs=True,
                                            file_has_mctruth_labels=True )
    entry = reader[0]
    print(entry.keys())

    cluster_labels = entry['cluster_labels'][0]
    point_pos = entry['shower_points']
    lms_filter = entry['lmshower_selection_mask'][0,:]

    print('point_pos: ',point_pos.shape)
    print('lms_filter: ',lms_filter.shape)
    print('instance_labels.shape: ',entry['instanceids'].shape)
    print('keypoint_labels, pre-filter: ',entry['kpscores'].shape)
    instance_labels = entry['instanceids'][0,lms_filter[:]]
    particle_labels = entry['particleids'][0,lms_filter[:]]

    keypoint_data = entry['keypoint_data']
    keypoint_labels = entry['kpscores'][:,lms_filter[:]]
    print('keypoint_labels, post-filter: ',keypoint_labels.shape)

    #llfile = "/home/twongjirad/working/data/mcc9_v40a_dl_run3b_NC_pi0_overlay_CV/merged_dlreco_mcc9_v40a_dl_run3b_NC_pi0_overlay_CV_aa444faa-530a-4fd7-b43f-b501bc221880.root"
    llfile = "/home/twongjirad/working/data/mcc9_v13_bnbnue_corsika/merged_dlreco_mcc9_v13_bnbnue_corsika_run00001_subrun00001.root"
    io = larlite.storage_manager(larlite.storage_manager.kREAD)
    io.add_in_filename( llfile )
    io.open()
    io.go_to(1)

    mcpg = ublarcvapp.mctools.MCPixelPGraph()
    mcpg.buildgraphonly( io )
    mcpg.printGraph(0,0)

    make_true_edge_list( cluster_labels, point_pos, instance_labels, particle_labels, keypoint_labels, keypoint_data, verbose=True, debug=False)
    
