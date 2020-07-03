#include "CosmicTrackBuilder.h"

#include "DataFormat/larflow3dhit.h"
#include "DataFormat/track.h"
#include "DataFormat/opflash.h"
#include "LArUtil/LArProperties.h"

#include "larflow/SCBoundary/SCBoundary.h"

namespace larflow {
namespace reco {

  void CosmicTrackBuilder::process( larcv::IOManager& iolcv,
                                    larlite::storage_manager& ioll )
  {

    // get clusters, pca-axis
    std::string producer = "trackprojsplit_full";
    
    larlite::event_larflowcluster* ev_cluster
      = (larlite::event_larflowcluster*)ioll.get_data(larlite::data::kLArFlowCluster, producer);
    larlite::event_pcaxis* ev_pcaxis
      = (larlite::event_pcaxis*)ioll.get_data(larlite::data::kPCAxis,producer);

    loadClusterLibrary( *ev_cluster, *ev_pcaxis );
    buildConnections();

    // get keypoints
    std::string producer_keypoint = "keypointcosmic";
    larlite::event_larflow3dhit* ev_keypoint
      = (larlite::event_larflow3dhit*)ioll.get_data(larlite::data::kLArFlow3DHit, producer_keypoint );


    for (size_t ikp=0; ikp<ev_keypoint->size(); ikp++) {
      auto const& kp = ev_keypoint->at(ikp);
      std::vector<float> startpt = { kp[0], kp[1], kp[2] };
      buildTracksFromPoint( startpt );
    }

    larlite::event_track* evout_track
      = (larlite::event_track*)ioll.get_data(larlite::data::kTrack, "cosmictrack");

    fillLarliteTrackContainer( *evout_track );

    if ( _do_boundary_analysis ) {
      _boundary_analysis( ioll );
    }
    
  }

  void CosmicTrackBuilder::_boundary_analysis( larlite::storage_manager& ioll )
  {
    // boundary analysis of tracks found.
    // we try 3 types of shifts to the boundary in order to estimate a t0 of the track
    // we use the boundary shifts to:
    //  (1) proposed t0 is compared to the flash times of opflashes, to look for possible matches
    //  (2) tag track as boundary muon, if both ends are near a boundary (through-going muon)

    const float drift_v = larutil::LArProperties::GetME()->DriftVelocity();
    
    // get the flash info
    std::vector< std::string > flash_producers_v = { "simpleFlashBeam",
                                                     "simpleFlashCosmic" };

    struct FlashInfo_t {
      std::string producer;
      int index;
      float usec;
      float anode_x;
      float cathode_x;
      float centroid_z;
    };

    std::vector< FlashInfo_t > flash_v;
    flash_v.reserve(100);
    
    for ( auto& prodname : flash_producers_v ) {
      larlite::event_opflash* ev_flash
        = (larlite::event_opflash*)ioll.get_data( larlite::data::kOpFlash, prodname );

      LARCV_INFO() << "opflashes from " << prodname << ": " << ev_flash->size() << std::endl;

      for (int idx=0; idx<(int)ev_flash->size(); idx++) {
        auto const& flash = ev_flash->at(idx);

        FlashInfo_t info;
        info.producer = prodname;
        info.index    = idx;
        info.usec     = flash.Time(); // usec from trigger
        // calculate x-position, if we assume track that made flash crossed anode
        // calculate x-position, if we assume track that made flash crossed cathode

        info.anode_x   = info.usec*drift_v;
        info.cathode_x = info.anode_x + 256.0;
        info.centroid_z = 0.;

        flash_v.push_back( info );
      }
    }

    LARCV_INFO() << "OpFlashes loaded: " << flash_v.size() << std::endl;


    // loop over reconstructed tracks
    larlite::event_track* ev_cosmic
      = (larlite::event_track*)ioll.get_data(larlite::data::kTrack,"cosmictrack");

    larflow::scb::SCBoundary scb;

    struct BoundaryMatch_t {
      int track_idx;
      int flash_idx;
      float minpt_dist2scb;
      float maxpt_dist2scb;
      float dist2scb; // min of the two above, for sorting
      bool operator<( const BoundaryMatch_t& rhs ) const {
        if ( dist2scb<rhs.dist2scb ) return true;
        return false;
      };
    };

    // save boundary matches
    std::vector<BoundaryMatch_t> track_boundary_v;
      
    for ( int itrack=0; itrack<(int)ev_cosmic->size(); itrack++ ) {

      auto const& track = ev_cosmic->at(itrack);

      LARCV_DEBUG() << "==== TRACK[" << itrack << "] ==========" << std::endl;
      LARCV_DEBUG() << "   start(" << track.Vertex()(0) << "," << track.Vertex()(1) << "," << track.Vertex()(2) << ") "
                    << "--> end(" << track.End()(0) << "," << track.End()(1) << "," << track.End()(2) << ") "
                    << std::endl;

      float x_min; 
      float x_max;
      float pt_min[3];
      float pt_max[3];
      if ( track.Vertex()(0)<track.End()(0) ) {
        x_min = track.Vertex()(0);
        x_max = track.End()(0);
        for (int v=0; v<3; v++) {
          pt_min[v] = track.Vertex()(v);
          pt_max[v] = track.End()(v);
        }
      }
      else {
        x_max = track.Vertex()(0);
        x_min = track.End()(0);
        for (int v=0; v<3; v++) {
          pt_max[v] = track.Vertex()(v);
          pt_min[v] = track.End()(v);
        }        
      }
      
      // we now pair with every flash, along with a null flash match where we shift
      // to the space-charge boundary in x
      int nmatches = 0;   //< number of flashes that if matched to track, moves it to a boundary
      int npossible = 0;  //< number of flashes that were consistent with track
      for (int iflash=0; iflash<(int)flash_v.size(); iflash++) {

        auto const& info = flash_v[iflash];

        float real_x_min = x_min-info.anode_x;
        float real_x_max = x_max-info.anode_x;

        //  we allow for slop of 10 cm;
        float x_offset = 0.;
        if ( real_x_min>-10 && real_x_min<0.0 ) {
          x_offset = -real_x_min;
        }
        else if ( real_x_max>256.0 && real_x_max<256.0+10.0 ) {
          x_offset = 256.0-real_x_max;
        }

        real_x_min += x_offset;
        real_x_max += x_offset;
          
        // check if track within flash bounds
        if ( real_x_min<0.0 || real_x_min>256.0
             || real_x_max<0.0 || real_x_max>256.0 ) {
          continue; // out-of-bounds
        }

        npossible++;
        
        // check distance to boundary of ends
        
        // min bound
        float dist2scb_min = scb.dist2boundary( real_x_min, pt_min[1], pt_min[2] );

        // max bound
        float dist2scb_max = scb.dist2boundary( real_x_max, pt_max[1], pt_max[2] );

        // ok, is anything close to the bound!
        if ( dist2scb_min<20.0 || dist2scb_max<20.0 ) {

          LARCV_DEBUG() << "qualifying boundary match: track[" << itrack << "]<->flash[" << iflash << "]" << std::endl;
          LARCV_DEBUG() << "   dist2scb@min extremum pt (" << real_x_min << "," << pt_min[1] << "," << pt_min[2] << "): "
                        << " " << dist2scb_min << std::endl;
          LARCV_DEBUG() << "   dist2scb@max extremum pt (" << real_x_max << "," << pt_max[1] << "," << pt_max[2] << "): "
                        << " " << dist2scb_max << std::endl;
          
          // record a boundary match!
          BoundaryMatch_t match;
          match.track_idx = itrack;
          match.flash_idx = iflash;
          match.minpt_dist2scb = dist2scb_min;
          match.maxpt_dist2scb = dist2scb_max;
          match.dist2scb = ( dist2scb_min<dist2scb_max ) ? dist2scb_min : dist2scb_max;
          track_boundary_v.push_back(match);
          nmatches++;
        }
        
      }//end of flash loop      

      LARCV_DEBUG() << "track[" << itrack << "] at bounds for " << nmatches << " flashes,"
                    << " of " << npossible << " flash-track matches" << std::endl;
    }//end of track loop
    
  }
  
}
}
