#include "NuTrackBuilder.h"
#include "larflow/Reco/cluster_functions.h"
#include "larcv/core/DataFormat/EventImage2D.h"

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
                                std::vector<NuVertexCandidate>& nu_candidate_v )
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
      larlite::event_track* ev_track
        = (larlite::event_track*)ioll.get_data(larlite::data::kTrack,producer);      
      loadClusterLibrary( *ev_cluster, *ev_pcaxis, *ev_track );
      
    }

    buildNodeConnections();
    
    set_output_one_track_per_startpoint( false );

    // wire plane images for getting dqdx later
    larcv::EventImage2D* ev_adc =
      (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D, "wire");
    auto const& adc_v = ev_adc->Image2DArray();

    // output containers
    larlite::event_track* evout_track
      = (larlite::event_track*)ioll.get_data(larlite::data::kTrack, "nutrack");
    larlite::event_track* evout_track_fitted
      = (larlite::event_track*)ioll.get_data(larlite::data::kTrack, "nutrack_fitted");
    
    for (auto& nuvtx : nu_candidate_v ) {

      nuvtx.track_v.clear();
      _track_proposal_v.clear(); // clear out track proposals

      LARCV_DEBUG() << "/////// [Vertex Start]: "
                    << "(" << nuvtx.pos[0] << "," << nuvtx.pos[1] << "," << nuvtx.pos[2] << ")"
                    << "/////////////"
                    << std::endl;

      // get nodes from each vertex
      std::vector<NodePos_t*> nodes_near_cluster_start;
      std::vector<NodePos_t*> nodes_far_cluster_start;      
      std::vector<int> segment_near_cluster_start;
      
      // loop over starting track clusters
      for ( auto const& vtxcluster : nuvtx.cluster_v ) {

        // only deal with tracks        
        if ( vtxcluster.type!=NuVertexCandidate::kTrack ) {
          nodes_near_cluster_start.push_back( nullptr );
          nodes_far_cluster_start.push_back( nullptr );          
          segment_near_cluster_start.push_back(-1);
          continue;
        }
        
        // veto nodes connected to the segment end closest to the vertexer
        int min_segidx = findClosestSegment( vtxcluster.pos, 20.0 );

        if ( min_segidx<0 ) {
          nodes_near_cluster_start.push_back( nullptr );
          nodes_far_cluster_start.push_back( nullptr );          
          segment_near_cluster_start.push_back(-1);          
          continue; // no matching segment
        }
        
        // operate the tracker to return all possible leaf paths
        // Nodes from the segment
        NodePos_t& node0 = _nodepos_v[2*min_segidx];
        NodePos_t& node1 = _nodepos_v[2*min_segidx+1];

        // determine which end of the segment is close to the vertex
        std::vector<float> enddist(2,0);
        for (int i=0; i<3; i++) {
          enddist[0] += ( nuvtx.pos[i] - node0.pos[i] )*( nuvtx.pos[i] - node0.pos[i] );
          enddist[1] += ( nuvtx.pos[i] - node1.pos[i] )*( nuvtx.pos[i] - node1.pos[i] );
        }

        NodePos_t* vtxnode   = ( enddist[0]<enddist[1] ) ? &node0 : &node1;
        NodePos_t* farnode   = ( enddist[0]<enddist[1] ) ? &node1 : &node0;
        
        nodes_near_cluster_start.push_back(vtxnode);
        nodes_far_cluster_start.push_back(farnode);
        segment_near_cluster_start.push_back(min_segidx);
      }//end of loop over vertex clusters
      
      // loop over starting track clusters
      int ivtx = -1;
      for ( auto const& vtxcluster : nuvtx.cluster_v ) {
        ivtx++;
        
        // only deal with tracks        
        if ( vtxcluster.type!=NuVertexCandidate::kTrack )
          continue;

        // get the cluster
        const larlite::larflowcluster& lfcluster
          = ((larlite::event_larflowcluster*)ioll.get_data(larlite::data::kLArFlowCluster, vtxcluster.producer))->at(vtxcluster.index);

        // transform back to cluster_t type
        larflow::reco::cluster_t cluster = larflow::reco::cluster_from_larflowcluster( lfcluster );
        
        LARCV_DEBUG() << "[NuTrackBuilder] Vertex Cluster seeding point: "
                      << "(" << vtxcluster.pos[0] << "," << vtxcluster.pos[1] << "," << vtxcluster.pos[2] << ")"
                      << std::endl;
        
        // reset the veto flags for the segment nodes
        TrackClusterBuilder::resetVetoFlags();

        // veto nodes connected to the segment end closest to the vertexer
        int min_segidx = segment_near_cluster_start[ivtx];

        if ( min_segidx<0 )
          continue; // no matching segment
        
        NodePos_t* vtxnode  = nodes_near_cluster_start[ivtx];

        // veto the other nodes
        for (int jvtx=0; jvtx<(int)nodes_near_cluster_start.size(); jvtx++) {
          if ( jvtx!=ivtx && nodes_near_cluster_start[jvtx] )
            nodes_near_cluster_start[jvtx]->veto = true;
          if ( nodes_far_cluster_start[jvtx] )
            nodes_far_cluster_start[jvtx]->veto = true;
        }
        vtxnode->veto = false;

        LARCV_DEBUG() << "[NuTrackBuilder] Vertex Cluster end near vertex: "
                      << "(" << vtxnode->pos[0] << "," << vtxnode->pos[1] << "," << vtxnode->pos[2] << ")"
                      << std::endl;
                       
        buildTracksFromPoint( vtxnode->pos );
        
      }//end of loop over vertex clusters


      // fill track containers for this 
      larlite::event_track unfitted_v;
      larlite::event_track fitted_v;
      larlite::event_larflowcluster fitted_hitcluster_v;
      larlite::event_larflowcluster unfitted_hitcluster_v;
      fillLarliteTrackContainer( unfitted_v, unfitted_hitcluster_v );
      fillLarliteTrackContainerWithFittedTrack( fitted_v, fitted_hitcluster_v, adc_v );
      LARCV_DEBUG() << "Vertex tracks: " << fitted_v.size() << "; hit clusters: " << fitted_hitcluster_v.size() << std::endl;

      // pass the fitted tracks to the nu candidate
      nuvtx.track_v.reserve( fitted_v.size() );
      for (auto& fitted : fitted_v )
        nuvtx.track_v.push_back(fitted);

      // pass the hit clusters on
      nuvtx.track_hitcluster_v.reserve( fitted_v.size() );
      for ( auto& hitcluster : fitted_hitcluster_v )
        nuvtx.track_hitcluster_v.emplace_back( std::move(hitcluster) );

      // pass the tracks to the output container
      for ( auto& unfitted : unfitted_v )
        evout_track->emplace_back( std::move(unfitted) );
      for ( auto& fitted: fitted_v )
        evout_track_fitted->emplace_back( std::move(fitted) );
      
    }//end of loop over vertex candidates

    LARCV_DEBUG() << "Number of tracks saved: " << evout_track_fitted->size() << std::endl;
  }
  
}
}
