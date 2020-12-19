#include "TripletTruthFixer.h"

#include  "larcv/core/DataFormat/EventImage2D.h"

namespace larflow {
namespace prep {

  void TripletTruthFixer::calc_reassignments( PrepMatchTriplets& tripmaker,
                                              larcv::IOManager& iolcv ) 
  {
    // start off by separately clustering points with different segment ids
    // then match clusters with mcshower objects using shower profile
    // these seed shower objects to which we assign instance labels

    // we can then choose to assign fragments of disconnect shower pieces back to the differe
    // shower trunks

    std::vector<int> pid_v;
    std::vector<larflow::reco::cluster_t> cluster_v;
    _cluster_same_showerpid_spacepoints( cluster_v, pid_v, tripmaker, true );


    // fix up proton id to not have such small clusters,
    // will reassign to ancestor or nearest proton id
    larcv::EventImage2D* ev_instance
      = (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D, "instance" );
    _reassignSmallTrackClusters( tripmaker, ev_instance->as_vector(), 10 );
    
    // associate shower cluster fragments
    // for each larlite mcshower and mctrack, we find closest trunk.
    // then we absorb fragments. save as graph

    // we need to relabel instance ids so that instances go from
    // [1,numinstances+1)
    
  }

  void TripletTruthFixer::_cluster_same_showerpid_spacepoints( std::vector<larflow::reco::cluster_t>& cluster_v,
                                                               std::vector<int>& pid_v,
                                                               larflow::prep::PrepMatchTriplets& tripmaker,
                                                               bool reassign_instance_labels )
  {
    cluster_v.clear();
    pid_v.clear();

    const float maxdist = 0.75;
    const int minsize = 10;
    const int maxkd = 50;
    
    std::vector<int> shower_ids = { 3, 4, 5 };
    for (int iid=0; iid<3; iid++) {
      int pid = shower_ids[iid];

      std::vector< std::vector<float> > point_v;
      std::vector< int > tripmaker_idx_v;
      point_v.reserve( tripmaker._pos_v.size() );
      tripmaker_idx_v.reserve( tripmaker._pos_v.size() );

      for ( int ipt=0; ipt<(int)tripmaker._pos_v.size(); ipt++) {
        if ( tripmaker._pdg_v[ipt]==pid ) {
          std::vector<float> pos = tripmaker._pos_v[ipt];
          point_v.push_back(pos);
          tripmaker_idx_v.push_back(ipt);
        }
      }

      std::vector< larflow::reco::cluster_t > pid_cluster_v;
      larflow::reco::cluster_sdbscan_spacepoints( point_v, pid_cluster_v, maxdist, minsize, maxkd );

      std::cout << "PID[" << pid << "] has " << pid_cluster_v.size() << " dbscan clusters" << std::endl;

      for (auto & cluster : pid_cluster_v ) {
        // for each cluster, we decide which label to use based on majority vote
        std::map<int,int> idcount;
        for ( auto const& hitidx : cluster.hitidx_v ) {
          int idx = tripmaker._instance_id_v[ tripmaker_idx_v[hitidx] ];
          if ( idcount.find(idx)==idcount.end() )
            idcount[idx] = 0;
          idcount[idx]++;
        }

        int maxcount = 0;
        int maxid = 0;
        for ( auto it=idcount.begin(); it!=idcount.end(); it++ ) {
          if ( maxcount<it->second ) {
            maxcount = it->second;
            maxid = it->first;
          }
        }

        // reassign the origina instance label.
        // also reassign the hit index to point back to triplet vector
        for ( auto & hitidx : cluster.hitidx_v ) {
          tripmaker._instance_id_v[ tripmaker_idx_v[hitidx] ] = maxid;
          hitidx =  tripmaker_idx_v[hitidx];
        }

        //xfer cluster to event container
        cluster_v.emplace_back( std::move(cluster) );
        pid_v.push_back( pid );
      }//end of loop over found clusters
      
    }//end of loop over pid
    
  }

  /**
   * @brief use neighboring pixels to reassign track clusters with small number of voxels
   *
   * These are often secondary protons or pions created by proton/pion reinteractions.
   * Small proton/pion will usually have some larger track with a proper instance nearby
   *
   */
  void TripletTruthFixer::_reassignSmallTrackClusters( larflow::prep::PrepMatchTriplets& tripmaker,
                                                       const std::vector< larcv::Image2D >& instanceimg_v,
                                                       const float threshold )
  {

    const int dvoxel = 3;
    const int nrows = instanceimg_v.front().meta().rows();
    const int ncols = instanceimg_v.front().meta().cols();    
    auto const& meta = instanceimg_v.front().meta(); // assuming metas the same for all planes

    // make list of track pixels to reassign: proton and charged pion labels
    std::vector<int> track_idx_v;
    // also make count of pixels in each instance
    std::map<int,int> track_instance_count; // key: instance id, value: count
    track_idx_v.reserve( tripmaker._pos_v.size() );
    for ( size_t i=0; i<tripmaker._pos_v.size(); i++ ) {
      if ( tripmaker._truth_v[i]==1 &&
           (tripmaker._pdg_v[i]>=7 && tripmaker._pdg_v[i]<=9) ) {
        track_idx_v.push_back(i);

        int instanceid = tripmaker._instance_id_v[i];
        if ( track_instance_count.find(instanceid)==track_instance_count.end() )
          track_instance_count[i] = 0;
        track_instance_count[i]++;
      }
    }

    // use this to decide which id should be used to replace
    // subthreshold instances
    struct ReplacementTally_t {
      int orig_instanceid;
      std::set<int> replacement_ids;
      ReplacementTally_t()
        : orig_instanceid(0) {};
      ReplacementTally_t(int id)
        : orig_instanceid(id) {};
    };
    std::map<int,ReplacementTally_t> replacement_tally_v;
    
    for ( size_t iidx=0; iidx<track_idx_v.size(); iidx++ ) {

      int triplet_idx = track_idx_v[iidx];

      std::vector<int> imgcoord = tripmaker.get_triplet_imgcoord_rowcol( triplet_idx );
      int truth_instance_index = tripmaker._instance_id_v[triplet_idx];
      
      auto it=track_instance_count.find( truth_instance_index );
      if ( it==track_instance_count.end() )
        continue; /// unexpected (should throw an error)
      
      if ( it->second>threshold )
        continue; // above threshold for reassignment
      
      auto it_t = replacement_tally_v.find( truth_instance_index );
      if ( it_t==replacement_tally_v.end() )
        replacement_tally_v[truth_instance_index] = ReplacementTally_t(truth_instance_index);

      auto& tally = replacement_tally_v[truth_instance_index];

      
      for (int dr=-dvoxel; dr<=dvoxel; dr++) {
        int row = (int)meta.row( imgcoord[3] ); // tick to row
        if (row<=0 || row>=nrows ) continue;
        
        for (int p=0; p<3; p++) {
          for (int dc=-dvoxel; dc<=dvoxel; dc++) {
            int col = imgcoord[p]+dc;
            if (col<0 || col>=ncols ) continue;

            int iid = instanceimg_v[p].pixel(row,col,__FILE__,__LINE__);

            // ignore own instanceid
            if ( iid==truth_instance_index || iid<0)
              continue;

            // this checks that its a track instance            
            if ( track_instance_count.find(iid)!=track_instance_count.end() )
              // tally track instance
              tally.replacement_ids.insert( iid );
            else {
              //std::cout << "trying to replace id=" << truth_instance_index << " but id=" << iid << " not in track count dict" << std::endl;
            }
          }//end of col loop
        }//end of plane loop
      }//end of row loop
    }
    
    // choose the replacement id
    // we pick the id associated to the largest track cluster
    std::map<int,int> replace_trackid;
    for ( auto it=track_instance_count.begin(); it!=track_instance_count.end(); it++ ) {
      if ( it->second>threshold)
        continue;
      
      auto& tally = replacement_tally_v[it->first];
      int max_nhits = 0;
      int max_replacement_id = 0;
      for (auto& id : tally.replacement_ids ) {
        if ( track_instance_count[id]>max_nhits ) {
          max_nhits = track_instance_count[id];
          max_replacement_id = id;
        }
      }
      replace_trackid[it->first] = max_replacement_id;
      if ( max_replacement_id>0 ) {
        std::cout << "TripletTruthFixer::_reassignSmallTrackClusters.L:" << __LINE__ << "] "
                  << "successfully replace trackid=" << it->first
                  << "( w/ " << it->second << " counts)"
                  << " with " << max_replacement_id << " (w/ " << max_nhits << ")"
                  << std::endl;
        track_instance_count[max_replacement_id] += it->second;
        it->second = 0;
        // need to propagate this reassignment to past reassignments
        for ( auto it_r=replace_trackid.begin(); it_r!=replace_trackid.end(); it_r++ ) {
          if ( it_r->second==it->first )
            it_r->second = max_replacement_id;
        }
      }
    }
    
    // execute the replacement
    int nreassigned = 0;
    for ( auto& tripidx : track_idx_v ) {
      int instanceid = tripmaker._instance_id_v[tripidx];      
      if ( replace_trackid.find( instanceid )!=replace_trackid.end() ) {
        tripmaker._instance_id_v[tripidx] = replace_trackid[instanceid];
        nreassigned++;
      }
    }
    std::cout << "[TripletTruthFixer::_reassignSmallTrackClusters] num reassigned = " << nreassigned << std::endl;
    
  }

  
}
}
