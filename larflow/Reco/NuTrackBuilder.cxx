#include "NuTrackBuilder.h"
#include "larflow/Reco/cluster_functions.h"

namespace larflow {
namespace reco {

  /**
   * @brief Run track builder on neutrino candidate tracks
   *
   * Using NuVertexCandidate instances as a seed, 
   * build out tracks by connecting cluster ends.
   *
   * @param[in] iolcv LArCV IO manager
   * @param[in] ioll  larlite IO manager
   * @param[in] nu_candidate_v Neutrino proto-vertices produced by NuVertexMaker.
   */
  void NuTrackBuilder::process( larcv::IOManager& iolcv,
                                larlite::storage_manager& ioll,
                                const std::vector<NuVertexCandidate>& nu_candidate_v )
  {

    // clear segments, connections, proposals
    clear();

    // get clusters, pca-axis
    std::vector< std::string > cluster_producers =
      { "trackprojsplit_full",
        "trackprojsplit_wcfilter" };
    
    for ( auto const& producer : cluster_producers ) {
    
      larlite::event_larflowcluster* ev_cluster
        = (larlite::event_larflowcluster*)ioll.get_data(larlite::data::kLArFlowCluster, producer);
      larlite::event_pcaxis* ev_pcaxis
        = (larlite::event_pcaxis*)ioll.get_data(larlite::data::kPCAxis,producer);      
      loadClusterLibrary( *ev_cluster, *ev_pcaxis );
      
    }
    
    buildNodeConnections();
    
    set_output_one_track_per_startpoint( false );

    for (auto const& nuvtx : nu_candidate_v ) {

      // loop over starting track clusters
      for ( auto const& vtxcluster : nuvtx.cluster_v ) {

        // only deal with tracks        
        if ( vtxcluster.type!=NuVertexCandidate::kTrack )
          continue;

        // get the cluster
        const larlite::larflowcluster& lfcluster
          = ((larlite::event_larflowcluster*)ioll.get_data(larlite::data::kLArFlowCluster, vtxcluster.producer))->at(vtxcluster.index);

        // transform back to cluster_t type
        larflow::reco::cluster_t cluster = larflow::reco::cluster_from_larflowcluster( lfcluster );
        
        // reset the veto flags for the segment nodes
        TrackClusterBuilder::resetVetoFlags();

        // veto nodes connected to the segment end closest to the vertexer
        int min_segidx = findClosestSegment( nuvtx.pos, 5.0 );

        if ( min_segidx<0 )
          continue; // no matching segment
        
        // operate the tracker to return all possible leaf paths
        // Nodes from the segment
        NodePos_t& node0 = _nodepos_v[2*min_segidx];
        NodePos_t& node1 = _nodepos_v[2*min_segidx+1];
        node0.inpath = true;
        node1.inpath = true;

        // determine which end of the segment is close to the vertex
        std::vector<float> enddist(2,0);
        for (int i=0; i<3; i++) {
          enddist[0] += ( nuvtx.pos[i] - node0.pos[i] )*( nuvtx.pos[i] - node0.pos[i] );
          enddist[1] += ( nuvtx.pos[i] - node1.pos[i] )*( nuvtx.pos[i] - node1.pos[i] );
        }

        NodePos_t* startnode = ( enddist[0]<enddist[1] ) ? &node1 : &node0;
        NodePos_t* vtxnode   = ( enddist[0]<enddist[1] ) ? &node0 : &node1;

        vtxnode->veto = true;
        
        //     LARCV_DEBUG() << " starting track from: (" << startnode->pos[0] << "," << startnode->pos[1] << "," << startnode->pos[2] << ")" << std::endl;
        
        //     // build paths
        //     auto it_segedge12 = _segedge_m.find( std::pair<int,int>(startnode->nodeidx,vtxnode->nodeidx) );
        //     auto it_segedge21 = _segedge_m.find( std::pair<int,int>(vtxnode->nodeidx,startnode->nodeidx) );    
        //     std::vector<float>& path_dir = it_segedge21->second.dir;
        
        //     std::vector< NodePos_t* > path;
        //     std::vector< const std::vector<float>* > path_dir_v;    
        //     std::vector< std::vector<NodePos_t*> > complete_v;
        
        
        
        //     // start at startnode
        //     path.clear();
        //     path_dir_v.clear();      
        //     path.push_back( startnode );
        //     path_dir_v.push_back( &path_dir );    
        //     _recursiveFollowPath( *startnode, path_dir, path, path_dir_v, complete_v );
        //     LARCV_DEBUG() << "[after start->next] point generated " << complete_v.size() << " possible tracks" << std::endl;
        // }
        
        buildTracksFromPoint( nuvtx.pos );
        
      }
    }

    larlite::event_track* evout_track
      = (larlite::event_track*)ioll.get_data(larlite::data::kTrack, "nutrack");

    fillLarliteTrackContainer( *evout_track );
    
  }
  
}
}
