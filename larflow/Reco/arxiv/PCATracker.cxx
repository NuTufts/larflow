#include "PCATracker.h"

#include <algorithm>

#include "DataFormat/pcaxis.h"
#include "DataFormat/track.h"
#include "geofuncs.h"
#include <cilantro/principal_component_analysis.hpp>

namespace larflow {
namespace reco {

  /**
   * @brief reconstruct tracks using event data in IO Managers
   *
   * 
   *
   * @param[in] iolcv LArCV IO manager
   * @param[in] ioll  larlite IO manager
   */
  void PCATracker::process( larcv::IOManager& iolcv, larlite::storage_manager& ioll )
  {

    // get larmatch hits
    larlite::event_larflow3dhit* ev_hit
      = (larlite::event_larflow3dhit*)ioll.get_data( larlite::data::kLArFlow3DHit, "maxtrackhit" );

    std::vector<int> used_v( ev_hit->size(), 0 );
    
    // make seeds
    _trackseeds_v.clear();
    _make_keypoint_seeds( ioll );
    
    // make seeds from cluster segments
    // _make_cluster_seeds( ioll );

    // sort seeds
    std::sort( _trackseeds_v.begin(), _trackseeds_v.end() );


    // output container
    larlite::event_track* evout_track =
      (larlite::event_track*)ioll.get_data( larlite::data::kTrack, "pcatracker" );
    
    for ( auto& seed : _trackseeds_v ) {
      std::vector<Segment_t> track = _build_track( seed, *ev_hit, used_v );
      if ( track.size()<=0 ) continue;

      // convert to larlite track      
      larlite::track lltrack;
      lltrack.reserve( track.size()+1 );
      for ( auto const& seg : track ) {
        lltrack.add_vertex( TVector3(seg.start_v[0], seg.start_v[1], seg.start_v[2]) );
        lltrack.add_direction( TVector3(seg.pca_v[0][0], seg.pca_v[0][1], seg.pca_v[0][2] ) );
      }
      lltrack.add_vertex(    TVector3(track.back().start_v[0], track.back().start_v[1], track.back().start_v[2]) );
      lltrack.add_direction( TVector3(track.back().pca_v[0][0], track.back().pca_v[0][1], track.back().pca_v[0][2]) );
      evout_track->emplace_back( std::move(lltrack) );
    }

    LARCV_INFO() << "Made " << evout_track->size() << "tracks" << std::endl;
  }
  

  /**
   * make StartSeed_t objects from keypoint data
   *
   */
  void PCATracker::_make_keypoint_seeds( larlite::storage_manager& ioll  )
  {

    larlite::event_larflow3dhit* ev_keypoint
      = (larlite::event_larflow3dhit*)ioll.get_data( larlite::data::kLArFlow3DHit, "keypoint_bigcluster" );

    larlite::event_pcaxis* ev_pcaxis
      = (larlite::event_pcaxis*)ioll.get_data( larlite::data::kPCAxis, "keypoint_bigcluster" );

    for ( size_t ikp=0; ikp<ev_keypoint->size(); ikp++ ) {
      auto const& kp   = ev_keypoint->at(ikp);
      auto const& axis = ev_pcaxis->at(ikp);

      StartSeed_t seed1;
      StartSeed_t seed2;      
      seed1.start = std::vector<float>( { kp[0], kp[1], kp[2] } );
      seed2.start = std::vector<float>( { kp[0], kp[1], kp[2] } );      
      seed1.pca_v.resize(3);
      seed2.pca_v.resize(3);      
      for (size_t v=0; v<3; v++) {
        
        seed1.pca_v[v] = std::vector<float>( { (float)axis.getEigenVectors()[v][0],
              (float)axis.getEigenVectors()[v][1],
              (float)axis.getEigenVectors()[v][2] } );
        seed1.eigenval_v = std::vector<float>( {(float)axis.getEigenValues()[0],
              (float)axis.getEigenValues()[1],
              (float)axis.getEigenValues()[2]} );

        seed2.pca_v[v] = std::vector<float>( { -(float)axis.getEigenVectors()[v][0],
              -(float)axis.getEigenVectors()[v][1],
              -(float)axis.getEigenVectors()[v][2] } );
        seed2.eigenval_v = std::vector<float>( {(float)axis.getEigenValues()[0],
              (float)axis.getEigenValues()[1],
              (float)axis.getEigenValues()[2]} );
      }//end of dim loop

      _trackseeds_v.emplace_back( std::move(seed1) );
      _trackseeds_v.emplace_back( std::move(seed2) );
    }
    
    LARCV_INFO() << "made " << _trackseeds_v.size() << " from keypoints" << std::endl;

    
  }//end of keypoint loop

  /**
   * build track from seed
   */
  std::vector< PCATracker::Segment_t > PCATracker::_build_track( const PCATracker::StartSeed_t& seed,
                                                                 const larlite::event_larflow3dhit& hit_v,
                                                                 std::vector<int>& used_v ) {
    std::vector< PCATracker::Segment_t > track;

    
    // we step, until we don't
    bool stop = false;
    std::vector<float> current_pos(3,0);
    std::vector<float> current_dir(3,0);

    // initialize with the seed
    current_pos = seed.start;
    current_dir = seed.pca_v[0];
    LARCV_DEBUG() << "build track with start "
                  << "pos=(" << current_pos[0] << "," << current_pos[1] << "," << current_pos[2] << ") "
                  << "dir=(" << current_dir[0] << "," << current_dir[1] << "," << current_dir[2] << ")"
                  << std::endl;
      
    while ( !stop ) {
      Segment_t seg = _get_next_segment(current_pos, current_dir, hit_v, used_v );
      if ( seg.seg_v.size()<=5 || seg.len<2.0 )
      stop = _decide_action( seg, hit_v, used_v ); ///< one day RL goes here!!!
      if ( !stop && seg.pca_v.size()>0 ) {
        for (int v=0; v<3; v++ ) {
          current_pos[v] = seg.end_v[v];
          current_dir[v] = seg.pca_v[0][v];
        }
        track.emplace_back( std::move(seg) );        
      }
    }
    LARCV_DEBUG() << "Track size: " << track.size() << std::endl;
    
    return track;
  }
  

  /**
   *
   *
   */
  PCATracker::Segment_t PCATracker::_get_next_segment( const std::vector<float> start,
                                                       const std::vector<float> start_dir,
                                                       const larlite::event_larflow3dhit& hit_v,
                                                       std::vector<int>& used_v )
  {

    // object to fill
    PCATracker::Segment_t seg;
    seg.start_v = start;
    seg.len = 10.;    
    // first we get points along the start direction for some distance X
    seg.end_v.resize(3,0);
    for (int v=0; v<3; v++) {
      seg.end_v[v] = start[v] + 10.0*start_dir[v];
    }

    float max_s = 0.0;
    for ( size_t idx=0; idx<hit_v.size(); idx++ ) {
      if ( used_v[idx]==1 ) continue;
      auto const& hit = hit_v[idx];

      // quick bounding box test
      bool inbox = true;
      for (int v=0; v<3; v++ ) {
        if ( hit[v]<seg.start_v[v]-5.0 || hit[v]>seg.end_v[v]+5.0 ) inbox = false;
        if ( !inbox ) break;
      }
      if ( !inbox ) continue;

      float s = pointRayProjection<float>( seg.start_v, start_dir, hit );
      if ( s<0 || s>10.0 ) continue;
      
      float r = pointLineDistance<float>( seg.start_v, seg.end_v, hit );
      if ( r>5.0 ) continue;
      
      SegmentHit_t seghit( r, s, idx );
      if ( s>max_s ) max_s = s;
      seg.seg_v.push_back( seghit );
      
    }
    LARCV_DEBUG() << "Collected " << seg.seg_v.size() << " hits for segment. max(s)=" << max_s << std::endl;

    if ( seg.seg_v.size()<=5 ) {
      LARCV_DEBUG() << "Too small return" << std::endl;
      return seg;
    }
    std::sort( seg.seg_v.begin(), seg.seg_v.end() );

    seg.pca_v.resize(3);
    seg.eigenval_v.resize(3,0);
    
    // get pca for different lengths
    float pcaratio = 10.;
    float seglen = max_s;
    max_s = 0.;
    while ( pcaratio>0.5 && seglen>1.0 ) {
      std::vector< Eigen::Vector3f > pt_v;
      float ms = 0;
      for ( auto const& seghit : seg.seg_v ) {
        if ( seghit.s>seglen ) break;
        if ( seghit.s>ms )
          ms = seghit.s;
        auto const& hit = hit_v[seghit.index];
        pt_v.push_back( Eigen::Vector3f( hit[0], hit[1], hit[2] ) );
      }
      if ( pt_v.size()>3 ) {        
        cilantro::PrincipalComponentAnalysis3f pca( pt_v );
        pcaratio = pca.getEigenValues()(1)/pca.getEigenValues()(0);
        for (int i=0; i<3; i++) {
          seg.eigenval_v[i] = pca.getEigenValues()(i);
          seg.pca_v[i] = std::vector<float>( { pca.getEigenVectors()(0,i),
                pca.getEigenVectors()(1,i),
                pca.getEigenVectors()(2,i) } );
        }

        LARCV_DEBUG() << "segment length=" << seglen << " has pcaratio=" << pcaratio << " max(s)=" << ms << " nhits=" << pt_v.size() << std::endl;
      }
      else {
        LARCV_DEBUG() << "no points" << std::endl;
        seg.pca_v.clear();
        seg.eigenval_v.clear();
        break;
      }
      
      if ( pcaratio>0.5 ) {
        seglen -= 2.0;
        // get ready to try again
      }
      else {
        // gonna keep this segment
        max_s = ms;
      }      
    }
    
    LARCV_DEBUG() << "Output of segment search: seglen=" << seglen << " pcaratio=" << pcaratio << " nhits=" << seg.seg_v.size() << std::endl;
    seg.len = max_s;
    for (int v=0; v<3; v++)
      seg.end_v[v] = seg.start_v[v] + seg.len*start_dir[v];

    return seg;
    
  }

  bool PCATracker::_decide_action( PCATracker::Segment_t& seg,
                                   const larlite::event_larflow3dhit& hit_v,
                                   std::vector<int>& used_v )
  {

    // determine stopping conditions
    
    if ( seg.seg_v.size()<5 )
      return true;

    if (seg.pca_v.size()==0 )
      return true;

    if ( seg.eigenval_v[0]/seg.eigenval_v[1] > 0.5 )
      return true;

    return false;
  }

}
}
