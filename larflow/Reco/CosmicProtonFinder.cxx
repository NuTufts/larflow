#include "CosmicProtonFinder.h"

#include "larlite/DataFormat/track.h"
#include "larlite/DataFormat/larflowcluster.h"
#include "TrackdQdx.h"

#include "cluster_functions.h"

namespace larflow {
namespace reco {

  CosmicProtonFinder::CosmicProtonFinder()
    : larcv::larcv_base("CosmicProtonFinder")
  {
    // set defaults
    _input_cosmic_treename_v.clear();
    _input_cosmic_treename_v.push_back( "boundarycosmicnoshift" );
    _input_cosmic_treename_v.push_back( "containedcosmic" );
    _output_tree_name = "cosmicproton";
  }
  
  /**
   * @brief process event data
   *
   */
  void CosmicProtonFinder::process( larcv::IOManager& iolcv,
                                    larlite::storage_manager& ioll )
  {

    LARCV_DEBUG() << "start" << std::endl;
    
    std::vector< larlite::larflowcluster > reclassified_cluster;
    std::vector< larlite::track > reclassified_track;

    // loop over cosmic tracks
    for ( auto const& cosmic_treename : _input_cosmic_treename_v ) {

      larlite::event_track* ev_track
        = (larlite::event_track*)ioll.get_data(larlite::data::kTrack, cosmic_treename);

      larlite::event_larflowcluster* ev_cluster
        = (larlite::event_larflowcluster*)ioll.get_data(larlite::data::kLArFlowCluster, cosmic_treename);

      LARCV_DEBUG() << cosmic_treename << " has " << ev_track->size() << " tracks and " << ev_cluster->size() << " clusters" << std::endl;
      
      int ntracks = ev_track->size();
      int num_reclassified = 0;
      std::vector<int> isproton(ntracks,0);

      for (int itrack=0; itrack<ntracks; itrack++) {
        
        auto& track = ev_track->at(itrack);
        auto& cluster = ev_cluster->at(itrack);

        int npts = track.NumberTrajectoryPoints();
        if ( npts<2 )
          continue;

        if ( track.NumberdQdx((larlite::geo::View_t)0)!=4 ) {
          LARCV_DEBUG() << "" << cosmic_treename << "[" << itrack << "] "
                        << "number of dqdx vectors (" << track.NumberdQdx((larlite::geo::View_t)0) << ") "
                        << "!= 4" << std::endl;
          continue;
        }
        
        // get length first
        float totpathlength = get_length(track);
        LARCV_DEBUG() << "check " << cosmic_treename << "[" << itrack << "]"
                      << " length=" << totpathlength << " cm"
                      << std::endl;

        if (totpathlength>50.0)
          continue;
        
        // if short enough, calculate likelihood
        LARCV_DEBUG() << "  calculate forward LL" << std::endl;
        float ll_forward  = _llpid.calculateLL( track, false );
        LARCV_DEBUG() << "  calculate backward LL" << std::endl;        
        float ll_backward = _llpid.calculateLL( track, true );

        // if consistent more w/ proton than muon, add to reclassify container        
        if ( ll_forward<0 ) {
          reclassified_cluster.push_back( cluster );
          reclassified_track.push_back( track );
          isproton[itrack] = 1;
          num_reclassified++;
        }
        else if ( ll_backward<0 ) {
          LARCV_DEBUG() << "  found proton, but need to reverse" << std::endl;          
          // have to reverse the track
          larlite::track rtrack;
          int npts = track.NumberTrajectoryPoints();
          rtrack.reserve( npts );
          std::vector<double> dqdx_v[4]; // for each view(s)
          for (int v=0; v<4; v++)
            dqdx_v[v].resize(npts,0);
          
          for (int ipt=npts-1; ipt>=0; ipt--) {
            rtrack.add_vertex( track.LocationAtPoint(ipt) );
            TVector3 rdir = track.DirectionAtPoint(ipt);
            for (int i=0; i<3; i++)
              rdir[i] *= -1.0;
            rtrack.add_direction( rdir );
            for (int v=0; v<4; v++)
              dqdx_v[v][ipt] = track.DQdxAtPoint(ipt,(larlite::geo::View_t)v);
          }
          for (int v=0; v<4; v++) 
            rtrack.add_dqdx( dqdx_v[v] );

          reclassified_cluster.push_back( cluster );
          reclassified_track.emplace_back( rtrack );
          isproton[itrack] = 1;
          num_reclassified++;
        }

        // (and remove from cosmic container ...)
        
      }//end of track loop of container

      // we need to remove reclassified track in vector
      if ( num_reclassified>0 ) {
        std::vector< larlite::track > still_cosmic_v;
        std::vector< larlite::larflowcluster > still_cluster_v;
        still_cosmic_v.reserve( ntracks );
        for (int itrack=0; itrack<ntracks; itrack++) {
          if ( isproton[itrack]==0 ) {
            still_cosmic_v.emplace_back( std::move(ev_track->at(itrack)) );
            still_cluster_v.emplace_back( std::move(ev_cluster->at(itrack)) );
          }
        }
        ev_track->clear();
        ev_cluster->clear();

        for (int itrack=0; itrack<(int)still_cosmic_v.size(); itrack++) {
          ev_track->emplace_back( std::move(still_cosmic_v[itrack]) );
          ev_cluster->emplace_back( std::move(still_cluster_v[itrack]) );        
        }
      }
      
      LARCV_DEBUG() << cosmic_treename << " now has " << ev_track->size() << " tracks and " << ev_cluster->size() << " clusters" << std::endl;
      LARCV_DEBUG() << "reclassified tracks: " << reclassified_track.size() << std::endl;
    }//loop over input containers

    LARCV_INFO() << "Found " << reclassified_cluster.size() << " proton-like tracks from cosmic tracks" << std::endl;


    
    // put reclassified clusters/tracks into output containers
    larlite::event_larflowcluster* evout_cluster
      = (larlite::event_larflowcluster*)ioll.get_data( larlite::data::kLArFlowCluster, _output_tree_name );
    larlite::event_track* evout_track
      = (larlite::event_track*)ioll.get_data( larlite::data::kTrack, _output_tree_name );
    larlite::event_pcaxis* evout_pcaxis
      = (larlite::event_pcaxis*)ioll.get_data( larlite::data::kPCAxis, _output_tree_name );

    for (int itrack=0; itrack<(int)reclassified_cluster.size(); itrack++) {

      auto& cluster = reclassified_cluster.at(itrack);
      larflow::reco::cluster_t c = larflow::reco::cluster_from_larflowcluster( cluster );
      larlite::pcaxis pca = larflow::reco::cluster_make_pcaxis( c, itrack );
      
      evout_cluster->emplace_back( std::move(reclassified_cluster.at(itrack)) );
      evout_track->emplace_back( std::move(reclassified_track.at(itrack)) );
      evout_pcaxis->emplace_back( std::move(pca) );
      
    }
    
  }

  float CosmicProtonFinder::get_length( const larlite::track& lltrack )
  {
    int npts = lltrack.NumberTrajectoryPoints();
    float totlen = 0.;
    for (int ipt=0; ipt<npts-1; ipt++) {
      auto const& pta = lltrack.LocationAtPoint(ipt);
      auto const& ptb = lltrack.LocationAtPoint(ipt+1);
      float seglen = (pta-ptb).Mag();
      totlen += seglen;
    }
    return totlen;
  }
  
}
}
