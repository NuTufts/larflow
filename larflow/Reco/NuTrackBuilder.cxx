#include "NuTrackBuilder.h"
#include "larflow/Reco/cluster_functions.h"
#include "larflow/Reco/TrackdQdx.h"
#include "larcv/core/DataFormat/EventImage2D.h"
#include "larlite/LArUtil/Geometry.h"
#include "ublarcvapp/RecoTools/DetUtils.h"

#include <ctime>
#include <algorithm>

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
                                std::vector<NuVertexCandidate>& nu_candidate_v,
				std::vector<ClusterBookKeeper>& nu_cluster_book_v,
				bool load_clusters )
  {

    std::clock_t t_start = std::clock();
    // We perform the reco one TPC at a time
    auto const geom = larlite::larutil::Geometry::GetME();
    for ( int icryo=0; icryo<(int)geom->Ncryostats(); icryo++) {
      for (int itpc=0; itpc<(int)geom->NTPCs(icryo); itpc++) {
	process_one_tpc( iolcv, ioll, nu_candidate_v, nu_cluster_book_v, itpc, icryo, true );

      }//end of TPC loop
    }//end of CRYO loop

    larlite::event_track* evout_track_fitted
      = (larlite::event_track*)ioll.get_data(larlite::data::kTrack, "nutrack_fitted");    
    LARCV_INFO() << "Number of tracks saved: " << evout_track_fitted->size() << std::endl;
    std::clock_t t_end = std::clock();
    float elapsed = float( t_end-t_start )/CLOCKS_PER_SEC;
    LARCV_INFO() << "end; elapsed=" << elapsed << " secs" << std::endl;

  }

  void NuTrackBuilder::process_one_tpc( larcv::IOManager& iolcv,
					larlite::storage_manager& ioll,
					std::vector<NuVertexCandidate>& nu_candidate_v,
					std::vector<ClusterBookKeeper>& nu_cluster_book_v,
					const int tpcid, const int cryoid,
					bool load_clusters )
  {  
    // ----------------------------------------------------------
    // INPUTS
    // ----------------------------------------------------------    
    larcv::EventImage2D* ev_adc =
      (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D, "wire");
    auto const& adc_v = ev_adc->Image2DArray();    
    
    // -----------------------------------------------------------
    // output containers
    // -----------------------------------------------------------
    larlite::event_track* evout_track
      = (larlite::event_track*)ioll.get_data(larlite::data::kTrack, "nutrack");
    larlite::event_track* evout_track_fitted
      = (larlite::event_track*)ioll.get_data(larlite::data::kTrack, "nutrack_fitted");

    LARCV_INFO() << "Build Tracks for Vertex Candidates (cryo,tpc)=(" << cryoid << "," << tpcid << ") ==========  " << std::endl;
	
    // clear segments, connections, proposals
    if ( load_clusters ) {
      clear_cluster_data();
      loadClustersAndConnections( iolcv, ioll, tpcid, cryoid );
    }
    clear_track_proposals();    
    set_output_one_track_per_startpoint( true );

    int nclusters = num_loaded_clusters();
    if ( nclusters==0 ) {
      LARCV_INFO() << "no clusters in this TPC" << std::endl;
      return;
    }
    
    // then building out tracks using NuVertexCandidates (vtx+clusters) as seeds
    for (size_t inuvertex=0; inuvertex<nu_candidate_v.size(); inuvertex++) {
      auto& nuvtx = nu_candidate_v.at(inuvertex);
      auto& vtx_cluster_book = nu_cluster_book_v.at(inuvertex);
      
      // only work on vertices inside the (CRYO,TPC)
      if ( nuvtx.cryoid!=cryoid || nuvtx.tpcid!=tpcid )
	continue;
      
      nuvtx.track_v.clear();
      _track_proposal_v.clear(); // clear out track proposals
      
      LARCV_DEBUG() << "/////// [Vertex Start]: "
		    << "(" << nuvtx.pos[0] << "," << nuvtx.pos[1] << "," << nuvtx.pos[2] << ")"
		    << " cryo=" << nuvtx.cryoid << " tpc=" << nuvtx.tpcid << " "
		    << "/////////////"
		    << std::endl;
      LARCV_DEBUG() << "  number of clusters: " << nuvtx.cluster_v.size() << std::endl;
      
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
	  LARCV_DEBUG() << "  * cluster not a track" << std::endl;
	  continue;
	}
        
	// veto nodes connected to the segment end closest to the vertexer
	int min_segidx = -1;
	try {
	  min_segidx = findClosestSegment( vtxcluster.pos, 20.0 );
	}
	catch ( const std::exception& e ) {
	  LARCV_CRITICAL() << "  * search for closest segment failed" << std::endl;
	  min_segidx = -1;
	  throw e;
	}
	
	if ( min_segidx<0 ) {
	  nodes_near_cluster_start.push_back( nullptr );
	  nodes_far_cluster_start.push_back( nullptr );          
	  segment_near_cluster_start.push_back(-1);
	  LARCV_DEBUG() << "  * track cluster has no close node " << std::endl;
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
	
	LARCV_DEBUG() << "  * setup nodes for track " << std::endl;
	
	nodes_near_cluster_start.push_back(vtxnode);
	nodes_far_cluster_start.push_back(farnode);
	segment_near_cluster_start.push_back(min_segidx);
      }//end of loop over vertex clusters
      
      LARCV_DEBUG() << "  -- start path search --" << std::endl;
      
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
        
	LARCV_DEBUG() << "Finding track paths from cluster seeding point: "
		      << "(" << vtxcluster.pos[0] << "," << vtxcluster.pos[1] << "," << vtxcluster.pos[2] << ")"
		      << std::endl;
	
	// reset the veto flags for the segment nodes
	TrackClusterBuilder::resetVetoFlags();
	
	// veto used clusters
	_veto_assigned_clusters( vtx_cluster_book );
	
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
	
	LARCV_DEBUG() << "==============================================" << std::endl;        
	LARCV_DEBUG() << "Start path search from vertex: "
		      << "(" << vtxnode->pos[0] << "," << vtxnode->pos[1] << "," << vtxnode->pos[2] << ")"
		      << std::endl;
	
	int nbefore = (int)_track_proposal_v.size();
	
	buildTracksFromPoint( vtxnode->pos );
	
	int nafter = (int)_track_proposal_v.size();
	
	_book_used_clusters( vtx_cluster_book );
	
	LARCV_DEBUG() << "After path search, number of proposals: " << nafter-nbefore << std::endl;
	LARCV_DEBUG() << "==============================================" << std::endl;
	
      }//end of loop over vertex clusters

      // fill track containers for this 
      larlite::event_track unfitted_v;
      larlite::event_track fitted_v;
      larlite::event_larflowcluster fitted_hitcluster_v;
      larlite::event_larflowcluster unfitted_hitcluster_v;
      fillLarliteTrackContainer( unfitted_v, unfitted_hitcluster_v );
      fillLarliteTrackContainerWithFittedTrack( fitted_v, fitted_hitcluster_v, adc_v );
      LARCV_DEBUG() << "Vertex tracks: " << fitted_v.size() << "; hit clusters: " << fitted_hitcluster_v.size() << std::endl;
      
      if ( fitted_v.size()!=fitted_hitcluster_v.size() ) {
	std::stringstream msg;
	msg << "num vertex tracks (" << fitted_v.size() << ") does not match num hit clusters (" << fitted_hitcluster_v.size() << ")" << std::endl;
	throw std::runtime_error( msg.str() );
      }
      
      // pass the fitted tracks to the nu candidate
      nuvtx.track_v.reserve( fitted_v.size() );
      std::vector<int> track_saved_v(fitted_v.size(),0);
      for (int itrack=0; itrack<(int)fitted_v.size(); itrack++) {
	auto& fitted = fitted_v.at(itrack);
	auto& hitcluster = fitted_hitcluster_v.at(itrack);
	
	// fitted with dqdx
	// larflow::reco::TrackdQdx dqdx_algo;
	// larlite::track track_dqdx;
	// try {
	//   track_dqdx = dqdx_algo.calculatedQdx( fitted, hitcluster, adc_v );
	//   if (track_dqdx.NumberTrajectoryPoints()>0) {
	//     track_saved_v[itrack] = 1;
	//     nuvtx.track_v.emplace_back( std::move(track_dqdx) );
	//   }
	// }
	// catch ( const std::exception& e ) {
	//   std::stringstream msg;
	//   msg << "error in trying to calculate dqdx track (id=" << itrack << "): " << e.what() << ". filling with original track" << std::endl;
	//   if (fitted.NumberTrajectoryPoints()>0) {
	//     track_saved_v[itrack] = 1;
	//     nuvtx.track_v.push_back(fitted);
	//   }
	// }
	if (fitted.NumberTrajectoryPoints()>0) {
	  track_saved_v[itrack] = 1;
	  nuvtx.track_v.push_back(fitted);
	}
      }
      
      // pass the hit clusters on
      nuvtx.track_hitcluster_v.reserve( fitted_v.size() );
      for ( int itrack=0; itrack<(int)fitted_hitcluster_v.size(); itrack++ ) {
	if ( track_saved_v[itrack]==0 )
	  continue;
	auto& hitcluster = fitted_hitcluster_v[itrack];
	nuvtx.track_hitcluster_v.emplace_back( std::move(hitcluster) );
      }
      
      // pass the tracks to the output container
      for ( auto& unfitted : unfitted_v )
	evout_track->emplace_back( std::move(unfitted) );
      for ( auto& fitted: fitted_v )
	evout_track_fitted->emplace_back( std::move(fitted) );
      
    }//end of loop over vertex candidates
    
  }

  void NuTrackBuilder::loadClustersAndConnections( larcv::IOManager& iolcv,
						   larlite::storage_manager& ioll,
						   const int tpcid,
						   const int cryoid )
  {

    std::clock_t t_start = std::clock();

    // get geometry info
    auto const geom = larlite::larutil::Geometry::GetME();
    
    // clear segments, connections, proposals
    clear();

    // wire plane images for getting dqdx later
    larcv::EventImage2D* ev_adc =
      (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D, "wire");
    auto const& adc_v = ev_adc->Image2DArray();

    // bad channel images for helping to determine proper gaps to jump
    larcv::EventImage2D* ev_badch =
      (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D, "badch" );
    auto const& badch_v = ev_badch->as_vector();
    
    // get images for this TPC and cryostat
    std::vector< const larcv::Image2D* > ptpc_adc_v
      = ublarcvapp::recotools::DetUtils::getTPCImages( adc_v, tpcid, cryoid );
    std::vector< const larcv::Image2D* > tpc_badch_v 
      = ublarcvapp::recotools::DetUtils::getTPCImages( badch_v, tpcid, cryoid );

    if ( ptpc_adc_v.size()==0 )
      return;
    
    // get clusters, pca-axis
    std::vector< std::string > cluster_producers =
      // { "trackprojsplit_full",
      //   "trackprojsplit_wcfilter",
      //   "hip" };
      { "trackprojsplit_wcfilter", "cosmicproton" };

    for ( auto const& producer : cluster_producers ) {

      LARCV_INFO() << "Adding clusters from '" << producer << "' producer/tree" << std::endl;
      larlite::event_larflowcluster* ev_cluster
        = (larlite::event_larflowcluster*)ioll.get_data(larlite::data::kLArFlowCluster, producer);
      larlite::event_pcaxis* ev_pcaxis
        = (larlite::event_pcaxis*)ioll.get_data(larlite::data::kPCAxis,producer);      
      larlite::event_track* ev_track
        = (larlite::event_track*)ioll.get_data(larlite::data::kTrack,producer);      
      loadClusterLibrary( *ev_cluster, *ev_pcaxis, *ev_track, tpcid, cryoid );
    }

    buildNodeConnections( &adc_v, &badch_v );

    std::clock_t t_end = std::clock();
    float elapsed = float( t_end-t_start )/CLOCKS_PER_SEC;
    LARCV_INFO() << "elapsed=" << elapsed << " secs" << std::endl;
    
  }

  void NuTrackBuilder::_veto_assigned_clusters( ClusterBookKeeper& nuvtx_cluster_book )
  {
    // we loop through the segments and associated nodes
    // the segment has a pointer to the cluster it represents
    // the cluster has its book-keeping index (in matchedflash_idx),
    // then we veto the segment's nodes
    //std::cout << "Start. Num of clusters in book: " << nuvtx_cluster_book.cluster_status_v.size() << std::endl;
    for ( auto& node : _nodepos_v ) {
      auto& seg = _segment_v.at(node.segidx);
      if ( seg.cluster ) {
	if ( seg.cluster->matchedflash_idx>=0
	     && seg.cluster->matchedflash_idx<(int)nuvtx_cluster_book.cluster_status_v.size() ) {
	  int status = nuvtx_cluster_book.cluster_status_v.at( seg.cluster->matchedflash_idx );
	  if ( status!=0 ) {
	    // used or marked
	    node.veto = true;
	  }
	}	  
      }
    }
    
  }

  void NuTrackBuilder::_book_used_clusters( ClusterBookKeeper& nuvtx_cluster_book )
  {
    int itrack = 0;
    for (auto const& track_proposal : _track_proposal_v ) {
      //std::cout << "itrack[" << itrack << "] ------" << std::endl;
      int nbooked = 0;
      for (  auto const& node : track_proposal ) {
	int clusterid = _segment_v.at( node->segidx ).cluster->matchedflash_idx;
	//std::cout << " node clusterid=" << clusterid << std::endl;
	if ( clusterid>=0 && clusterid<nuvtx_cluster_book.cluster_status_v.size() ) {
	  nuvtx_cluster_book.cluster_status_v.at(clusterid) = 1;
	  nbooked++;
	}
      }
      //std::cout << "num of clusters booked: " << nbooked << std::endl;
      itrack++;
    }
  }
}
}
