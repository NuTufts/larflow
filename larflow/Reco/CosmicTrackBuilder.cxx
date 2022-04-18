#include "CosmicTrackBuilder.h"

#include "larlite/DataFormat/larflow3dhit.h"
#include "larlite/DataFormat/track.h"
#include "larlite/DataFormat/opflash.h"
#include "larlite/LArUtil/LArProperties.h"
#include "larlite/LArUtil/Geometry.h"

#include "larcv/core/DataFormat/EventImage2D.h"

#include "larflow/SCBoundary/SCBoundary.h"
#include "TrackdQdx.h"

namespace larflow {
namespace reco {

  CosmicTrackBuilder::CosmicTrackBuilder()
    : _do_boundary_analysis(true),
      _using_default_cluster(true)
  {
    _cluster_tree_v.clear();
    _cluster_tree_v.push_back( "trackprojsplit_full" );
    producer_keypoint = "keypointcosmic";      
  }
  
  /**
   * @brief Process the event data in the IO managers
   *
   * The data expected in the IO managers are:
   * @verbatim embed:rst:lead-asterisk
   *  * larflowcluster of track clusters
   *  * corresponding principle components for the track clusters
   *  * keypoints stored in the form of larflow3dhit(s)
   * @endverbatim
   *
   * The output will be a container of larlite::track instances.
   * @verbatim embed:rst:lead-asterisk
   *  * larlite::tracks of muons that touch the space charge boundary 
   *    and are shifted in 'x' to be on the boudnary. [default tree name: boundarycosmic]
   *  * larlite::tracks of muons that tourch the space charge boundary [defaul: boundarycosmicnoshift ]
   *  * larlite::tracks of contained muons [ default: containedcosmic ]
   * @endverbatim
   *
   * @param[in] iolcv LArCV IO manager where we get event data
   * @param[in] ioll  larlite IO manager where we get data and store outputs
   *
   */
  void CosmicTrackBuilder::process( larcv::IOManager& iolcv,
                                    larlite::storage_manager& ioll )
  {

    clear();
    
    for ( auto const& producer : _cluster_tree_v ) {
      larlite::event_larflowcluster* ev_cluster
        = (larlite::event_larflowcluster*)ioll.get_data(larlite::data::kLArFlowCluster, producer);
      larlite::event_pcaxis* ev_pcaxis
        = (larlite::event_pcaxis*)ioll.get_data(larlite::data::kPCAxis,producer);
      larlite::event_track* ev_track
        = (larlite::event_track*)ioll.get_data(larlite::data::kTrack,producer);
      loadClusterLibrary( *ev_cluster, *ev_pcaxis, *ev_track );
    }

    // wire plane images for getting dqdx later
    larcv::EventImage2D* ev_adc =
      (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D, "wire");
    auto const& adc_v = ev_adc->Image2DArray();
    
    // bad channel images for helping to determine proper gaps to jump
    larcv::EventImage2D* ev_badch =
      (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D, "badch" );
    auto const& badch_v = ev_badch->as_vector();
    
    buildNodeConnections( &adc_v, &badch_v );

    // make tracks using keypoints
    larlite::event_larflow3dhit* ev_keypoint
      = (larlite::event_larflow3dhit*)ioll.get_data(larlite::data::kLArFlow3DHit, producer_keypoint );


    for (size_t ikp=0; ikp<ev_keypoint->size(); ikp++) {
      auto const& kp = ev_keypoint->at(ikp);
      std::vector<float> startpt = { kp[0], kp[1], kp[2] };
      buildTracksFromPoint( startpt );
    }

    // make tracks using unused segments
    _buildTracksFromSegments();
    
    larlite::event_track* evout_track
      = (larlite::event_track*)ioll.get_data(larlite::data::kTrack, "cosmictrack");
    larlite::event_larflowcluster* evout_trackcluster
      = (larlite::event_larflowcluster*)ioll.get_data(larlite::data::kLArFlowCluster, "cosmictrack");

    fillLarliteTrackContainerWithFittedTrack( *evout_track, *evout_trackcluster, adc_v );

    // apply dqdx calc to fitted track
    for (int itrack=0; itrack<(int)evout_track->size(); itrack++) {
      auto& fitted = evout_track->at(itrack);
      auto& hitcluster = evout_trackcluster->at(itrack);
      
      larflow::reco::TrackdQdx dqdx_algo;
      larlite::track track_dqdx;
      try {
        track_dqdx = dqdx_algo.calculatedQdx( fitted, hitcluster, adc_v );
        // swap it
        std::swap(fitted,track_dqdx);
      }
      catch ( const std::exception& e ) {
        std::stringstream msg;
        msg << "error in trying to calculate dqdx track (id=" << itrack << "): " << e.what() << "." << std::endl;
      }
    }
    
    larlite::event_track* evout_simpletrack
      = (larlite::event_track*)ioll.get_data(larlite::data::kTrack, "simplecosmictrack");
    larlite::event_larflowcluster* evout_simpletrackcluster
      = (larlite::event_larflowcluster*)ioll.get_data(larlite::data::kLArFlowCluster, "simplecosmictrack");    
    fillLarliteTrackContainer( *evout_simpletrack, *evout_simpletrackcluster ); 

    if ( _do_boundary_analysis ) {
      _boundary_analysis_noflash( ioll );
    }
    
  }

  /**
   * @brief  boundary analysis of reconstructed tracks 
   * 
   * we try 3 types of shifts to the boundary in order to estimate a t0 of the track
   * we use the boundary shifts to:
   * @verbatim embed:rst:leading-asterisk
   *  1. proposed t0 is compared to the flash times of opflashes, to look for possible matches
   *  2. tag track as boundary muon, if both ends are near a boundary (through-going muon)
   * @endverbatim
   *
   *
   * @param[in] ioll IO manager to store output tracks
   *
   */
  void CosmicTrackBuilder::_boundary_analysis_noflash( larlite::storage_manager& ioll )
  {

    const float drift_v = larutil::LArProperties::GetME()->DriftVelocity();
    
    // loop over reconstructed tracks
    larlite::event_track* ev_cosmic
      = (larlite::event_track*)ioll.get_data(larlite::data::kTrack,"cosmictrack");
    larlite::event_larflowcluster* ev_cosmic_hitcluster
      = (larlite::event_larflowcluster*)ioll.get_data(larlite::data::kLArFlowCluster,"cosmictrack");

    // class to calculate distance to space charge boundary
    larflow::scb::SCBoundary scb;

    struct BoundaryAna_t {
      int track_idx;
      int num_ends_on_boundary;
      float x_offset;
    };

    // save boundary matches, one set per track
    std::vector<BoundaryAna_t> track_boundary_v(ev_cosmic->size());

    const float contained_dist = 10;
      
    for ( int itrack=0; itrack<(int)ev_cosmic->size(); itrack++ ) {

      auto const& track = ev_cosmic->at(itrack);

      LARCV_DEBUG() << "==== TRACK[" << itrack << "] ==========" << std::endl;
      LARCV_DEBUG() << "start(" << track.Vertex()(0) << "," << track.Vertex()(1) << "," << track.Vertex()(2) << ") "
                    << "--> end(" << track.End()(0) << "," << track.End()(1) << "," << track.End()(2) << ") "
                    << std::endl;
      
      BoundaryAna_t& ba = track_boundary_v.at(itrack);
      ba.track_idx = itrack;
      ba.num_ends_on_boundary = 0;
      ba.x_offset = 0.;
      
      // mark out-of-time track
      if ( track.Vertex()(0)<0.0 || track.Vertex()(0)>256.0
           || track.End()(0)<0.0 || track.End()(0)>256.0 ) {
        ba.num_ends_on_boundary = 1;
        LARCV_DEBUG() << "track is out-of-time" << std::endl;
        continue;
      }

      // check non anode/cathode boundaries
      if ( 116.0-fabs(track.Vertex()(1))<contained_dist
           || track.Vertex()(2)<contained_dist
           || track.Vertex()(2)>1036-contained_dist
           || 116.0-fabs(track.End()(1))<contained_dist
           || track.End()(2)<contained_dist
           || track.End()(2)>1036.0-contained_dist )  {
        ba.num_ends_on_boundary = 1;
        LARCV_DEBUG() << "track is already near TPC boundary" << std::endl;
        continue;
      }

      // calculate x-position if we shift point to the cathode SCE
      float xboundary_min = scb.XatBoundary( track.Vertex()(0), track.Vertex()(1), track.Vertex()(2) );
      float xboundary_max = scb.XatBoundary( track.End()(0), track.End()(1), track.End()(2) );

      float xshift_min = xboundary_min-track.Vertex()(0);
      float xshift_max = xboundary_max-track.End()(0);

      // if we move the min point, check the max-point
      int btype_min = -1;
      float dwall_min_at_scb = scb.dist2boundary( track.End()(0)+xshift_min, track.End()(1), track.End()(2), btype_min );
      bool min_at_scb_valid  = (track.End()(0)+xshift_min)>=0 && (track.End()(0)+xshift_min)<=256.0;

      // if we move the max point, check the min-point is on the space charge boundary
      int btype_max = -1;
      float dwall_max_at_scb = scb.dist2boundary( track.Vertex()(0)+xshift_max, track.Vertex()(1), track.Vertex()(2), btype_max );
      bool max_at_scb_valid  = (track.Vertex()(0)+xshift_max)>=0 && (track.Vertex()(0)+xshift_max)<=256.0;
      
      LARCV_DEBUG() << " if is shifted to cathode SCB: " << std::endl;
      LARCV_DEBUG() << "    move min: (" << track.Vertex()(0)+xshift_min << "," <<  track.Vertex()(1) << "," << track.Vertex()(2) << ") <--> "
                    << "(" << track.End()(0)+xshift_min << "," <<  track.End()(1) << "," <<  track.End()(2) << ")"
                    << " dist=" << dwall_min_at_scb << " b=" << btype_min
                    << std::endl;
      LARCV_DEBUG() << "    move max: (" << track.Vertex()(0)+xshift_max << "," <<  track.Vertex()(1) << "," << track.Vertex()(2) << ") <--> "
                    << "(" << track.End()(0)+xshift_max << "," <<  track.End()(1) << "," <<  track.End()(2) << ")"
                    << " dist=" << dwall_max_at_scb << " b=" << btype_max
                    << std::endl;
      
      // does one of them work
      if ( (fabs(dwall_min_at_scb)<contained_dist && min_at_scb_valid )
           || (fabs(dwall_max_at_scb)<contained_dist && max_at_scb_valid ) ) {
        ba.num_ends_on_boundary = 1;

        if ( min_at_scb_valid && max_at_scb_valid ) {
          if ( fabs(dwall_min_at_scb)<fabs(dwall_max_at_scb) )
            ba.x_offset = xshift_min;
          if ( fabs(dwall_min_at_scb)<fabs(dwall_max_at_scb) )
            ba.x_offset = xshift_max;
        }
        else if ( min_at_scb_valid && !max_at_scb_valid ) {
          ba.x_offset = xshift_max;
        }
        else if ( !min_at_scb_valid && max_at_scb_valid ) {
          ba.x_offset = xshift_min;
        }
        
        LARCV_DEBUG() << "is thrumu if moved to SCB" << std::endl;
        continue;
      }

      // move min to anode
      int btype_anode_min = -1;
      float dwall_min_at_anode = scb.dist2boundary( track.End()(0)-track.Vertex()(0), track.End()(1), track.End()(2), btype_anode_min );
      bool min_at_anode_valid  = (track.End()(0)-track.Vertex()(0))>=0.0 && (track.End()(0)-track.Vertex()(0))<=256;
      
      // move max to anode
      int btype_anode_max = -1;
      float dwall_max_at_anode = scb.dist2boundary( track.Vertex()(0)-track.End()(0), track.Vertex()(1), track.Vertex()(2), btype_anode_max );
      bool max_at_anode_valid  = (track.Vertex()(0)-track.End()(0))>=0.0 && (track.Vertex()(0)-track.End()(0))<=256;

      LARCV_DEBUG() << "  track if shifted to anode: " << std::endl;
      LARCV_DEBUG() << "    move min: (" << 0 << "," <<  track.Vertex()(1) << "," << track.Vertex()(2) << ") <--> "
                    << "(" << track.End()(0)-track.Vertex()(0) << "," <<  track.End()(1) << "," <<  track.End()(2) << ")"
                    << " dist2_scb=" << dwall_min_at_anode << " btype=" << btype_anode_min
                    << std::endl;
      LARCV_DEBUG() << "    move max: (" << track.Vertex()(0)-track.End()(0) << "," <<  track.Vertex()(1) << "," << track.Vertex()(2) << ") <--> "
                    << "(" << 0.0 << "," <<  track.End()(1) << "," <<  track.End()(2) << ")"
                    << " dist2_scb=" << dwall_max_at_anode << " btype=" << btype_anode_max
                    << std::endl;
      
      if ( (fabs(dwall_min_at_anode)<2.0 && min_at_anode_valid )
           || (fabs(dwall_max_at_anode)<2.0 && max_at_anode_valid ) ) {
        ba.num_ends_on_boundary = 1;

        if ( min_at_anode_valid && max_at_anode_valid ) {
          if ( fabs(dwall_min_at_anode)<fabs(dwall_max_at_anode) )
            ba.x_offset = xshift_min;
          if ( fabs(dwall_min_at_anode)<fabs(dwall_max_at_anode) )
            ba.x_offset = xshift_max;
        }
        else if ( min_at_anode_valid && !max_at_anode_valid ) {
          ba.x_offset = xshift_max;
        }
        else if ( !min_at_anode_valid && max_at_anode_valid ) {
          ba.x_offset = xshift_min;
        }        
        
        LARCV_DEBUG() << "is thrumu if moved to anode" << std::endl;
        continue;
      }

      LARCV_DEBUG() << "track is contained" << std::endl;
      
    }//end of loop over track

    // save track with matches along with flash as a pair
    larlite::event_track* evout_boundary_track =
      (larlite::event_track*)ioll.get_data(larlite::data::kTrack,"boundarycosmic");
    larlite::event_track* evout_boundary_noshift_track =
      (larlite::event_track*)ioll.get_data(larlite::data::kTrack,"boundarycosmicnoshift");
    larlite::event_track* evout_contained_track =
      (larlite::event_track*)ioll.get_data(larlite::data::kTrack,"containedcosmic");
    larlite::event_larflowcluster* evout_boundary_noshift_cluster =
      (larlite::event_larflowcluster*)ioll.get_data(larlite::data::kLArFlowCluster,"boundarycosmicnoshift");
    larlite::event_larflowcluster* evout_contained_cluster =
      (larlite::event_larflowcluster*)ioll.get_data(larlite::data::kLArFlowCluster,"containedcosmic");

    for ( auto const& bm : track_boundary_v ) {
      auto const& track = ev_cosmic->at(bm.track_idx);
      auto const& clust = ev_cosmic_hitcluster->at(bm.track_idx);
      larlite::track cptrack;
      cptrack.reserve( track.NumberTrajectoryPoints() );
      for ( int ipt=0; ipt<(int)track.NumberTrajectoryPoints(); ipt++ ) {
        const TVector3& pos = track.LocationAtPoint(ipt);
        cptrack.add_vertex( TVector3(pos(0)+bm.x_offset,pos(1),pos(2)) );
        cptrack.add_direction( track.DirectionAtPoint(ipt) );
      }

      if ( bm.num_ends_on_boundary==1 ) {
        evout_boundary_track->emplace_back( std::move(cptrack) );
        evout_boundary_noshift_track->push_back( track );
        evout_boundary_noshift_cluster->push_back( clust );
      }
      else {
        evout_contained_track->emplace_back( std::move(cptrack) );
        evout_contained_cluster->push_back( clust );
      }
    }

    LARCV_DEBUG() << "input cosmics=" << ev_cosmic->size() << "; "
                  << "boundary=" << evout_boundary_track->size() << "; "
                  << "contained=" << evout_contained_track->size()
                  << std::endl;
    
  }

  /**
   * @brief boundary analysis of tracks found, using flash
   *
   * [deprecated]
   *
   * @param[in] ioll larlite IO manager
   * 
   */
  void CosmicTrackBuilder::_boundary_analysis_wflash( larlite::storage_manager& ioll )
  {
    // 

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
      float centroid_y;
      float totpe;
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
        info.centroid_y = 0.;
        info.totpe = 0.;
        std::vector<float> pe_v(32,0);
        for (int iopdet=0; iopdet<(int)flash.nOpDets(); iopdet++) {
          float pe = flash.PE(iopdet);
          if ( pe<=0 ) {
            continue;
          }
          int ch = iopdet%32;
          if ( pe>pe_v[ch] && ch<32)
            pe_v[ch] = pe;
        }
        for (int ch=0; ch<32; ch++) {
          float pe = pe_v[ch];
          std::vector<double> xyz;
          //larutil::Geometry::GetME()->GetOpDetPosition( ch, xyz );
          larutil::Geometry::GetME()->GetOpChannelPosition( ch, xyz );
          info.totpe += pe;
          info.centroid_z += pe*xyz[2];
          info.centroid_y += pe*xyz[1];
        }
        if ( info.totpe>0 ) {
          info.centroid_z /= info.totpe;
          info.centroid_y /= info.totpe;
        }
        LARCV_DEBUG() << "FLASH[" << flash_v.size() << "] pe=" << info.totpe << " centroid-z=" << info.centroid_z << std::endl;
        
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
      float x_offset;
      float x_min;
      float x_max;
      float minpt_dist2scb;
      float maxpt_dist2scb;
      int minpt_btype;
      int maxpt_btype;
      float dist2scb; // min of the two above, for sorting
      int num_ends_on_boundary;
      float pmt_dz;
      bool operator<( const BoundaryMatch_t& rhs ) const {
        if ( dist2scb<rhs.dist2scb ) return true;
        return false;
      };
    };

    // save boundary matches, one set per track
    std::vector< std::vector<BoundaryMatch_t> > track_boundary_vv(ev_cosmic->size());
      
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

      float z_min;
      float z_max;
      if ( track.Vertex()(2)<track.End()(2) ) {
        z_min = track.Vertex()(2);
        z_max = track.End()(2);
      }
      else {
        z_max = track.Vertex()(2);
        z_min = track.End()(2);
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
        int btype_min = -1;
        float dist2scb_min = scb.dist2boundary( real_x_min, pt_min[1], pt_min[2], btype_min );

        // max bound
        int btype_max = -1;
        float dist2scb_max = scb.dist2boundary( real_x_max, pt_max[1], pt_max[2], btype_max );

        // ok, is anything close to the bound!
        if ( (dist2scb_min<10.0 || dist2scb_max<10.0 ) && btype_min!=btype_max ) {


          // record a boundary match!
          BoundaryMatch_t match;
          match.track_idx = itrack;
          match.flash_idx = iflash;
          match.x_offset = -info.anode_x+x_offset;
          match.x_min = real_x_min;
          match.x_max = real_x_max;
          match.minpt_dist2scb = dist2scb_min;
          match.maxpt_dist2scb = dist2scb_max;
          match.minpt_btype = btype_min;
          match.maxpt_btype = btype_max;
          match.dist2scb = fabs(dist2scb_min)+fabs(dist2scb_max);

          if ( info.centroid_z<z_min )
            match.pmt_dz = info.centroid_z-z_min;
          else if ( info.centroid_z>z_max )
            match.pmt_dz = z_max - info.centroid_z;
          else {
            match.pmt_dz = fabs(0.5*(z_min+z_max) - info.centroid_z);
          }
         
          match.num_ends_on_boundary = 0;
          if ( dist2scb_min<3.0 ) match.num_ends_on_boundary++;
          if ( dist2scb_max<3.0 ) match.num_ends_on_boundary++;

          LARCV_DEBUG() << "qualifying boundary match: track[" << itrack << "]<->flash[" << iflash << "]" << std::endl;
          LARCV_DEBUG() << "   dist2scb@min extremum pt (" << real_x_min << "," << pt_min[1] << "," << pt_min[2] << "): "
                        << " " << dist2scb_min << std::endl;
          LARCV_DEBUG() << "   dist2scb@max extremum pt (" << real_x_max << "," << pt_max[1] << "," << pt_max[2] << "): "
                        << " " << dist2scb_max << std::endl;
          LARCV_DEBUG() << "  pmt_dz: " << match.pmt_dz << std::endl;
          
          
          track_boundary_vv[itrack].push_back(match);
          nmatches++;
        }
        
      }//end of flash loop      

      LARCV_DEBUG() << "track[" << itrack << "] at bounds for " << nmatches << " flashes,"
                    << " of " << npossible << " flash-track matches" << std::endl;
    }//end of track loop


    // now we make boundary tag + flash assignments

    // first we go through tracks where one end x-position is > 100 cm. this is range where being
    //  on the boundary and matching to a flash is valuable info!
    // we first make assignment to tracks with only one 2-boundary matches
    // we then go through others and veto matches to previously matched flashes
    //   of the non-veto'd matches, we pick the ones closest to the boundary

    std::vector< BoundaryMatch_t > final_match_v;
    std::vector<int> flash_used_v( flash_v.size(), 0 );
    std::vector<int> tracks_matched_v( track_boundary_vv.size(), 0 );

    // first pass: accept matches where there is only one solution
    //             one end must be > 100 cm in position
    for ( size_t itrack=0; itrack<track_boundary_vv.size(); itrack++ ) {

      // first, count number of qualifying matches
      int nmatches = 0;
      const BoundaryMatch_t* qualifying_match = nullptr;
      for ( auto const& bm : track_boundary_vv[itrack] ) {

        if ( (bm.x_min>100.0 || bm.x_max>100.0) && bm.dist2scb<3.0 && bm.num_ends_on_boundary==2 ) {
          qualifying_match = &bm;
          nmatches++;
        }
        
      }

      if ( nmatches==1 ) {
        // accept this match
        final_match_v.push_back( *qualifying_match );
        flash_used_v[ qualifying_match->flash_idx ] = 1;
        tracks_matched_v[ itrack ] = 1;
        LARCV_DEBUG() << "FIRST PASS MATCH: track[" << itrack << "] <--> flash[" << qualifying_match->flash_idx << "]" << std::endl;
      }
      
    }//end of track loop

    LARCV_DEBUG() << "Number of first pass matches: "
                  << final_match_v.size() << " tracks of " << track_boundary_vv.size() << std::endl;

    // second pass, 2 ends matched, best match viz smallest dist2scb, do not reuse first pass flashes.
    int n2ndpass = 0;
    for ( size_t itrack=0; itrack<track_boundary_vv.size(); itrack++ ) {    
      if ( tracks_matched_v[itrack]==1 ) continue;

      // loop through matches, accept only 2-end, exclude previous flash matches
      // rank match by pmt_dz for qualifying matches only
      const BoundaryMatch_t* min_match = nullptr;
      float min_match_dist = -1e9;
      for ( size_t ibm=0; ibm<track_boundary_vv[itrack].size(); ibm++ ) {
        auto const& bm = track_boundary_vv[itrack].at(ibm);
        if( flash_used_v[ bm.flash_idx ]==1 ) continue;
        
        if ( bm.dist2scb<5.0 && bm.num_ends_on_boundary==2 && min_match_dist<bm.pmt_dz) {
          min_match_dist = bm.pmt_dz;
          min_match = &bm;
        }        
      }

      if ( min_match ) {
        final_match_v.push_back( *min_match );
        flash_used_v[ min_match->flash_idx ] = 2;
        tracks_matched_v[itrack] = 2;
        n2ndpass++;
        LARCV_DEBUG() << "SECOND PASS MATCH: track[" << itrack << "] <--> flash[" << min_match->flash_idx << "]" << std::endl;
      }

    }

    LARCV_DEBUG() << "Number of second pass matches: "
                  << n2ndpass << ", total matches: " << final_match_v.size() << " tracks of " << track_boundary_vv.size()
                  << std::endl;

    // third pass: 1 end matched, best match viz smallest dist2scb of matched end, do not reuse first+second pass flashes.
    int n3rdpass = 0;
    for ( size_t itrack=0; itrack<track_boundary_vv.size(); itrack++ ) {    
      if ( tracks_matched_v[itrack]>0 ) continue;

      // loop through matches, accept only 2-end, exclude previous flash matches
      const BoundaryMatch_t* min_match = nullptr;
      float min_match_dist = 1e9;
      for ( size_t ibm=0; ibm<track_boundary_vv[itrack].size(); ibm++ ) {
        auto const& bm = track_boundary_vv[itrack].at(ibm);
        if( flash_used_v[ bm.flash_idx ]>0 ) continue;
        
        if ( bm.num_ends_on_boundary==1 ) {

          float minend_dist = (bm.minpt_dist2scb<bm.maxpt_dist2scb) ? bm.minpt_dist2scb : bm.maxpt_dist2scb;
          if( minend_dist<5.0 && minend_dist<min_match_dist) {
            min_match_dist = minend_dist;
            min_match = &bm;
          }
        }        
      }

      if ( min_match ) {
        final_match_v.push_back( *min_match );
        flash_used_v[ min_match->flash_idx ] = 3;
        tracks_matched_v[itrack] = 3;
        n3rdpass++;
        LARCV_DEBUG() << "THIRD PASS MATCH: track[" << itrack << "] <--> flash[" << min_match->flash_idx << "]" << std::endl;                
      }

    }//end of third pass track loop
    LARCV_DEBUG() << "Number of third pass matches: "
                  << n3rdpass << ", total matches: " << final_match_v.size() << " tracks of " << track_boundary_vv.size()
                  << std::endl;

    // save track with matches along with flash as a pair
    larlite::event_track* evout_match_track =
      (larlite::event_track*)ioll.get_data(larlite::data::kTrack,"matchedcosmictrack");
    larlite::event_opflash* evout_match_opflash =
      (larlite::event_opflash*)ioll.get_data(larlite::data::kOpFlash,"matchedcosmictrack");

    for ( auto const& bm : final_match_v ) {
      auto const& track = ev_cosmic->at(bm.track_idx);
      larlite::track cptrack;
      cptrack.reserve( track.NumberTrajectoryPoints() );
      for ( int ipt=0; ipt<(int)track.NumberTrajectoryPoints(); ipt++ ) {
        const TVector3& pos = track.LocationAtPoint(ipt);
        cptrack.add_vertex( TVector3(pos(0)+bm.x_offset,pos(1),pos(2)) );
        cptrack.add_direction( track.DirectionAtPoint(ipt) );
      }
      
      evout_match_track->emplace_back( std::move(cptrack) );

      larlite::event_opflash* ev_flash
        = (larlite::event_opflash*)ioll.get_data( larlite::data::kOpFlash, flash_v[bm.flash_idx].producer );
      evout_match_opflash->push_back( ev_flash->at(flash_v[bm.flash_idx].index) );

    }
    
  }

  void CosmicTrackBuilder::add_cluster_treename( std::string treename )
  {
    if ( _using_default_cluster ) {
      // we are going to replace the default cluster tree, so clear it out.
      _using_default_cluster = false;
      _cluster_tree_v.clear();
    }
    _cluster_tree_v.push_back( treename );
  }
  
}
}
