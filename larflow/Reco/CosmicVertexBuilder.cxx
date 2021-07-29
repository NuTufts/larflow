#include "CosmicVertexBuilder.h"

#include "larcv/core/DataFormat/EventImage2D.h"
#include "DataFormat/track.h"
#include "DataFormat/larflowcluster.h"
#include "LArUtil/LArProperties.h"
#include "LArUtil/Geometry.h"
#include "ublarcvapp/Reco3D/TrackReverser.h"

#include "larflow/LArFlowConstants/LArFlowConstants.h"
#include "TrackdQdx.h"

namespace larflow {
namespace reco {

  /**
   * @brief process event data to find stopping muons
   *
   * we find this sample for calibration purposes.
   * 
   * start by seeding possible vertices using
   * @verbatim embed:rst:leading-asterisk
   *  * keypoints 
   *  * intersections of particle clusters (not yet implemented)
   *  * vertex activity near ends of partice clusters (not yet implemented)
   * @endverbatim
   *
   * output:
   * @verbatim embed:rst:leading-asterisk
   *  * vertex candidates stored in _vertex_v
   *  * need to figure out way to store in larcv or larlite iomanagers
   * @endverbatim
   *
   * @param[in] iolcv Instance of LArCV IOManager with event data
   * @param[in] ioll  Instance of larlite storage_manager containing event data
   *
   */
  void CosmicVertexBuilder::process( larcv::IOManager& iolcv,
                                     larlite::storage_manager& ioll,
                                     std::vector<NuVertexCandidate>& nu_candidate_v )
  {

    // we combine cosmic tracks with shower cluster points
    const float max_dist = 10.0;

    // get cosmic track containers. made by cosmictrackbuilder.
    larlite::event_track* ev_boundary_noshift_track =
      (larlite::event_track*)ioll.get_data(larlite::data::kTrack,"boundarycosmicnoshift");
    larlite::event_track* ev_contained_track =
      (larlite::event_track*)ioll.get_data(larlite::data::kTrack,"containedcosmic");

    larlite::event_larflowcluster* ev_boundary_noshift_cluster =
      (larlite::event_larflowcluster*)ioll.get_data(larlite::data::kLArFlowCluster,"boundarycosmicnoshift");
    larlite::event_larflowcluster* ev_contained_cluster =
      (larlite::event_larflowcluster*)ioll.get_data(larlite::data::kLArFlowCluster,"containedcosmic");


    // get shower clusters
    larlite::event_larflowcluster* ev_shower_cluster_v
      = (larlite::event_larflowcluster*)ioll.get_data( larlite::data::kLArFlowCluster, "showerkp" );
    larlite::event_pcaxis* ev_shower_pca_v
      = (larlite::event_pcaxis*)ioll.get_data( larlite::data::kPCAxis, "showerkp" );

    // get adc image
    larcv::EventImage2D* ev_adc
      = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, "wire" );
    auto const& adc_v = ev_adc->as_vector();


    std::vector< larlite::event_track* > ev_track_v
      = { ev_boundary_noshift_track, ev_contained_track };
    
    std::vector< larlite::event_larflowcluster* > ev_trackcluster_v
      = { ev_boundary_noshift_cluster, ev_contained_cluster };

    std::vector<larflow::reco::NuVertexCandidate> candidate_v;
    std::vector< TVector3 > past_pos_v; /// location of past candidates. use to avoid making repeats.
    
    for (size_t icontainer=0; icontainer<ev_track_v.size(); icontainer++) {
      
      auto& pev_track   = ev_track_v.at(icontainer);
      auto& pev_cluster = ev_trackcluster_v.at(icontainer);

      for (size_t itrack=0; itrack<pev_track->size(); itrack++) {

        auto const& track = pev_track->at(itrack);
        auto const& trackcluster = pev_cluster->at(itrack);

        // check to see if we pair with any showers
        TVector3 start = track.LocationAtPoint(0);
        TVector3 end   = track.LocationAtPoint( (int)track.NumberTrajectoryPoints()-1 );
        
        for ( size_t icluster=0; icluster<ev_shower_cluster_v->size(); icluster++ ) {
          
          auto const& lfcluster = ev_shower_cluster_v->at(icluster);
          auto const& lfpca     = ev_shower_pca_v->at(icluster);

          if ( lfpca.getEigenVectors().size()<5 )
            continue;
          
          const std::vector<double>& v1 = lfpca.getEigenVectors()[3];
          const std::vector<double>& v2 = lfpca.getEigenVectors()[4];

          float dist[4] = {0};
          for (int i=0; i<3; i++) {
            dist[0] += (v1[i]-start[i])*(v1[i]-start[i]);
            dist[1] += (v1[i]-end[i])*(v1[i]-end[i]);
            dist[2] += (v2[i]-start[i])*(v2[i]-start[i]);
            dist[3] += (v2[i]-end[i])*(v2[i]-end[i]);
          }

          float mindist = 1e9;
          int minindex = -1;
          for (int j=0; j<4; j++) {
            if ( dist[j]<mindist ) {
              mindist = dist[j];
              minindex = j;
            }
          }

          LARCV_DEBUG() << "testing track[" << icontainer << "," << itrack << "] and shower[" << icluster << "]: mindist=" << mindist << std::endl;

          if ( mindist>max_dist )
            continue;
              
          //make a candidate
          larflow::reco::NuVertexCandidate cosmicvtx;

          cosmicvtx.keypoint_producer = "cosmicvertexbuilder";
          cosmicvtx.keypoint_index    = (int)candidate_v.size();
          cosmicvtx.keypoint_type     = (int)larflow::kStopMuVtx;
          cosmicvtx.pos.resize(3,0);
          TVector3 posv;
          if ( minindex==0 || minindex==2 ) {
            for (int i=0; i<3; i++) 
              cosmicvtx.pos[i] = start[i];
            posv = start;
          }
          else {
            for (int i=0; i<3; i++)
              cosmicvtx.pos[i] = end[i];
            posv = end;
          }

          float min_prev_dist = 1e9;
          // check for overlap. O(N^2) check, but we dont anticipate having many candidates.
          for ( auto const& past_pos : past_pos_v ) {
            float past_dist = (posv-past_pos).Mag();
            if ( past_dist < min_prev_dist )
              min_prev_dist = past_dist;
          }
          if ( min_prev_dist<max_dist ) {
            LARCV_DEBUG() << "New candidate too close to previous candidate: " << min_prev_dist << " cm" << std::endl;
            continue;
          }

          // new candidate

          cosmicvtx.tick = 3200 + posv[0]/larutil::LArProperties::GetME()->DriftVelocity()/0.5;
          if ( cosmicvtx.tick > adc_v.at(0).meta().min_y()
               && cosmicvtx.tick < adc_v.at(0).meta().max_y() ) {
            cosmicvtx.row = adc_v.at(0).meta().row( cosmicvtx.tick );
          }
          else {
            // invalid row
            continue;
          }

          // checks over, we're going to save this candiate.
          
          // add to lis tof past candidate positions
          past_pos_v.push_back( posv );
          
          size_t ndqdx_pts = track.NumberdQdx((larlite::geo::View_t)0);
          if ( ndqdx_pts!=4 ) {
            LARCV_DEBUG() << " need to make track dqdx as information is missing" << std::endl;
            larflow::reco::TrackdQdx dqdx_algo;
            larlite::track wdqdx = dqdx_algo.calculatedQdx( track, trackcluster, adc_v );
            LARCV_DEBUG() << " after trackdqdx: numdqdx=" << wdqdx.NumberdQdx((larlite::geo::View_t)0) << std::endl;
            cosmicvtx.track_v.push_back(wdqdx);
          }
          else {
            // has dqdx info
            cosmicvtx.track_v.push_back( track );
          }
          cosmicvtx.track_hitcluster_v.push_back( trackcluster );
          if ( minindex==1 || minindex==3 ) {
            // need to reverse track
            LARCV_DEBUG() << " need to reverse track direction" << std::endl;
            larlite::track reverse = ublarcvapp::reco3d::TrackReverser::reverseTrack( cosmicvtx.track_v.back() );
            LARCV_DEBUG() << " num dqdx_v after reverse: " << reverse.NumberdQdx((larlite::geo::View_t)0) << std::endl;
            std::swap( cosmicvtx.track_v.back(), reverse );
          }
          
          cosmicvtx.col_v.resize(3,0);
          for (int p=0; p<3; p++) {
            cosmicvtx.col_v[p] = larutil::Geometry::GetME()->WireCoordinate( posv, p );
          }
          cosmicvtx.score = 0.;
          
          cosmicvtx.shower_v.push_back( lfcluster );
          cosmicvtx.shower_pcaxis_v.push_back( lfpca );
          
          larlite::track shower_trunk;
          TVector3 shstart;
          TVector3 shend;
          if ( minindex<2) {
            // same shower dir as PCA
            for (int i=0; i<3; i++) {
              shstart[i] = v1[i];
              shend[i]   = v2[i];
            }
          }
          else {
            // reverse shower dir from PCA
            for (int i=0; i<3; i++) {
              shstart[i] = v2[i];
              shend[i]   = v1[i];
            }
          }
          TVector3 shower_dir = shend-shstart;
          double showerlen = shower_dir.Mag();
          if ( showerlen>0 ) {
            for (int i=0; i<3; i++)
              shower_dir[i] /= showerlen;
          }
          shower_trunk.add_direction(shower_dir);
          shower_trunk.add_direction(shower_dir);
          shower_trunk.add_vertex(shstart);
          shower_trunk.add_vertex(shend);            
          cosmicvtx.shower_trunk_v.push_back( shower_trunk );
          
          LARCV_DEBUG() << "STORING VERTEX CANDIDATE!" << std::endl;
          
          // add candidate
          candidate_v.emplace_back( std::move(cosmicvtx) );
          
        }//end of shower list
        
      }//end of track list      
    }//end of container loop
    
    LARCV_INFO() << "Saving " << candidate_v.size() << " cosmic vertex candidates" << std::endl;
    for ( auto& cand : candidate_v ) {
      nu_candidate_v.emplace_back( std::move(cand) );
    }
  }
  
}
}
