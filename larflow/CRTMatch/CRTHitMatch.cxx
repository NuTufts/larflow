#include "CRTHitMatch.h"

#include "LArUtil/LArProperties.h"
#include "larflow/Reco/cluster_functions.h"

#include "TRandom2.h"

namespace larflow {
namespace crtmatch {

  void CRTHitMatch::clear() {

    // input containers
    _intime_opflash_v.clear();
    _outtime_opflash_v.clear();
    _crthit_v.clear();
    _crttrack_v.clear();
    _lfcluster_v.clear();
    _pcaxis_v.clear();
    _hit2track_rank_v.clear();
    _all_rank_v.clear();

    // output containers
    _matched_hitidx.clear();
    _flash_v.clear();
    _match_hit_v.clear();
    _matched_cluster.clear();
    _matched_cluster_t.clear();    
    _matched_opflash_v.clear();
    _used_tracks_v.clear();
    
  }
  
  void CRTHitMatch::addIntimeOpFlashes( const larlite::event_opflash& opflash_v ) {
    for ( auto const& opf : opflash_v )
      _intime_opflash_v.push_back( opf );
  }

  void CRTHitMatch::addCosmicOpFlashes( const larlite::event_opflash& opflash_v ) {
    for ( auto const& opf : opflash_v )
      _outtime_opflash_v.push_back( opf );
  }

  void CRTHitMatch::addCRThits( const larlite::event_crthit& crthit_v ) {
    for ( auto const& hit : crthit_v )
      _crthit_v.push_back( hit );
  }
  
  void CRTHitMatch::addCRTtracks( const larlite::event_crttrack& crttrack_v ) {
    for ( auto const& track : _crttrack_v )
      _crttrack_v.push_back( track );
  }
  
  void CRTHitMatch::addLArFlowClusters( const larlite::event_larflowcluster& lfcluster_v,
                                        const larlite::event_pcaxis& pcaxis_v ) {
    for ( auto const& cluster : lfcluster_v ) {
      _lfcluster_v.push_back( &cluster );
    }
    
    for ( auto const& pca : pcaxis_v ) {
      _pcaxis_v.push_back( &pca );
    }
  }

  bool CRTHitMatch::process( larcv::IOManager& iolcv, larlite::storage_manager& llio ) {

    clear();

    larlite::event_opflash* beamflash
      = (larlite::event_opflash*)llio.get_data( larlite::data::kOpFlash, "simpleFlashBeam" );
    larlite::event_opflash* cosmicflash
      = (larlite::event_opflash*)llio.get_data( larlite::data::kOpFlash, "simpleFlashCosmic" );
    addIntimeOpFlashes( *beamflash );
    addCosmicOpFlashes( *cosmicflash );

    // get crt hits
    larlite::event_crthit* crthit_v
      = (larlite::event_crthit*)llio.get_data( larlite::data::kCRTHit, "crthitcorr" );
    addCRThits( *crthit_v );

    // get clusters
    larlite::event_larflowcluster* lfclusters_v
      = (larlite::event_larflowcluster*)llio.get_data( larlite::data::kLArFlowCluster, "pcacluster" );
    larlite::event_pcaxis* pcaxis_v
      = (larlite::event_pcaxis*)llio.get_data( larlite::data::kPCAxis, "pcacluster" );
    addLArFlowClusters( *lfclusters_v, *pcaxis_v );

    std::vector< larflow::reco::cluster_t > test_v;
    int icluster = 0;
    for ( auto const& plfcluster : _lfcluster_v ) {
      std::cout << "[convert cluster " << icluster  << "]" << std::endl;
      larflow::reco::cluster_t c = larflow::reco::cluster_from_larflowcluster( *plfcluster );
      test_v.emplace_back( std::move(c) );
      icluster++;
    }
    LARCV_INFO() << "passed conversion test" << std::endl;
    
    return makeMatches();
  }
  
  bool CRTHitMatch::makeMatches() {

    // compile matches
    compilematches();

    _used_tracks_v.resize( _lfcluster_v.size(), 0 );

    // process matches in greedy fashion
    for ( auto& m : _all_rank_v ) {

      // get index of crt hit of this match
      int crthitidx = m.hitidx;

      auto& hit_matches_v = _hit2track_rank_v[ crthitidx ];

      if ( hit_matches_v.size()==0 ) continue; // shouldn't happen
      
      if ( hit_matches_v.size()==1 ) {

        // if track already used. skip
        if ( _used_tracks_v[ m.trackidx ]==1 )
          continue;

        // if not, make the match
        LARCV_INFO() << " store single crt-hit to cluster match products" << std::endl;
        _matched_hitidx.push_back( crthitidx );
        // create a copy
        larlite::larflowcluster  lfc   = *_lfcluster_v[ m.trackidx ]; // create a copy
        _matched_cluster.emplace_back( std::move(lfc) );
        _used_tracks_v[ m.trackidx ] = 1;

      }
      else {
        // more than 1, attempt a merge

        // set the usage vector for these matches
        bool has_unused = false;
        std::vector<int> used_track_v( hit_matches_v.size(), 0 );
        for ( size_t i=0; i<hit_matches_v.size(); i++ ) {
          used_track_v[i] = _used_tracks_v[ hit_matches_v[i].trackidx ];
          if ( used_track_v[i]==0 )
            has_unused = true;
        }

        if ( has_unused ) {
          bool performed_merge = false;          
          larlite::larflowcluster merge = _merge_matched_cluster( hit_matches_v,
                                                                  used_track_v,
                                                                  performed_merge );

          // make match based on merge result
          if (performed_merge) {
            larflow::reco::cluster_t clout = larflow::reco::cluster_from_larflowcluster( merge );            
            _matched_hitidx.push_back( crthitidx );
            _matched_cluster.push_back( merge );
            _used_tracks_v[ m.trackidx ] = 1;          
            LARCV_INFO() << "store merged clusters" << std::endl;
          }
        }

      }
      
      
    }//end of loop over all rank matches

    LARCV_NORMAL() << "number of crt-hit to track matches: " << _matched_cluster.size() << std::endl;

    // apply t0 offset to clusters
    for ( size_t imatch=0; imatch<_matched_hitidx.size(); imatch++ ) {
      int hitidx = _matched_hitidx[ imatch ];
      auto const& crthit = _crthit_v[ hitidx ];
      auto&  lfc = _matched_cluster[ imatch ];
      float xoffset = larutil::LArProperties::GetME()->DriftVelocity()*( crthit.ts2_ns*0.001 );
      // we need to t0 subtract the points
      for ( auto& lfhit : lfc ) {
        lfhit[0] -= xoffset;
      }
      // store cluster_t for modified larflow cluster
      larflow::reco::cluster_t clout = larflow::reco::cluster_from_larflowcluster( lfc );
      _matched_cluster_t.emplace_back( std::move(clout) );      
    }

    // now match CRT hits with tracks to opflashes
    // make list of flashes
    for ( auto const& flash : _intime_opflash_v )
      _flash_v.push_back( &flash );
    for ( auto const& flash : _outtime_opflash_v )
      _flash_v.push_back( &flash );
    // make list of crt hits with matches
    for ( auto const& hitidx : _matched_hitidx ) {
      _match_hit_v.push_back( &_crthit_v[ hitidx ] );
    }
    _matchOpflashes( _flash_v, _match_hit_v, _matched_cluster, _matched_opflash_v );

    return true;
    
  }

  /**
   * run track match to crt hits
   *
   * use add functions to provide inputs. need the following:
   *
   * larlite products:
   *   opflash: typically 'simpleFlashBeam' and 'simpleFlashCosmic'
   *   larflowcluster: typically 'pcacluster'
   *   pcaxis: pc axis to larflowcluster above, typically 'pcacluster'
   *   crttrack: typically 'crttrack'
   *   crthit: typically 'crthitcorr'
   *  
   */
  void CRTHitMatch::compilematches() {

    _hit2track_rank_v.resize( _crthit_v.size() );

    // plane positions
    float crt_plane_pos[4] = { -261.606, -142.484, 393.016, 618.25 }; // bottom, side, side, top

    //larlite::event_pcaxis* ev_pcaout = (larlite::event_pcaxis*)outio.get_data( larlite::data::kPCAxis, "crtmatch" );

    // loop over larflow tracks, assign it to the best hit in each CRT plane
    // we are filling out _hit2track_rank_v[ hit index ] -> list of track indices
    LARCV_INFO() << "==============================" << std::endl;
    LARCV_INFO() << " Start match function" << std::endl;
    if ( _pcaxis_v.size()!=_lfcluster_v.size() ) {
      throw std::runtime_error( "number of clusters and pc axis do not match" );
    }
    
    for (size_t itrack=0; itrack<_pcaxis_v.size(); itrack++ ) {

      // represent track and its direction with its pc axis
      auto const& pca = *_pcaxis_v[itrack];

      // enforce minimum length
      float len = getLength(pca);
      if ( len<5.0 ) continue;

      // allocate vars for best intersection for each CRT plane
      float min_dist[4]  = { 1.0e9, 1.0e9, 1.0e9, 1.0e9 };
      int best_hitidx[4] = {-1, -1, -1, -1};
      std::vector<float> best_panel_pos[4];
      for (int i=0; i<4; i++) best_panel_pos[i].resize(3,0);

      // loop over crt hits
      for (size_t jhit=0; jhit<_crthit_v.size(); jhit++) {
        auto const& crthit = _crthit_v[jhit];
        int crtplane = crthit.plane;

        // calculate distance to crt hit and position of intersection on the panel
        std::vector<float> panel_pos;
        float dist = makeOneMatch( pca, crthit, panel_pos );

        // update closest hit on the plane
        if ( dist>0 && dist<min_dist[crtplane] ) {
          min_dist[crtplane] = dist;
          best_hitidx[crtplane] = jhit;
          best_panel_pos[crtplane] = panel_pos;
        }
      }//end of hit loop
      LARCV_DEBUG() << " [" << itrack << "] closest dist per plane = "
                    << "[ " << min_dist[0] << ", " << min_dist[1] << ", " << min_dist[2] << ", " << min_dist[3] << "]"
                    << std::endl;

      // loop over plane
      for (int p=0; p<4; p++ ) {

        // if a good hit found in the plane
        if ( best_hitidx[p]>=0 ) {
          auto const& besthit = _crthit_v[best_hitidx[p]];        
          LARCV_DEBUG() << " crt_hit=(" << besthit.x_pos << ", " << besthit.y_pos << "," << besthit.z_pos << ") "
                        << " panel_pos=(" << best_panel_pos[p][0] << "," << best_panel_pos[p][1] << "," << best_panel_pos[p][2] << ") "
                        << std::endl;

          if ( min_dist[p]>=0 && min_dist[p]<50.0 ) {
            float line[3] = {0};
            float dist = 0.;
            for (int i=0; i<3; i++ ) {
              line[i] = best_panel_pos[p][i]-pca.getAvePosition()[i];
              dist += line[i]*line[i];
            }
            dist = sqrt(dist);
            for (int i=0; i<3; i++ ) line[i] /= dist;
            
            // store a match
            match_t m;
            m.hitidx   = best_hitidx[p];
            m.trackidx = itrack;
            m.tracklen = len;
            m.dist2hit = min_dist[p];
            
            _hit2track_rank_v[ best_hitidx[p] ].push_back( m );
            _all_rank_v.push_back( m );
            
            // store for visualization purposes
            // larlite::pcaxis::EigenVectors e_v; // 3-axis + 2-endpoints
            // for ( size_t p=0; p<3; p++ ) {
            //   std::vector<double> da_v = { line[0], line[1], line[2] };
            //   e_v.push_back( da_v );
            // }
            // std::vector<double> centroid_v = { pca.getAvePosition()[0], pca.getAvePosition()[1], pca.getAvePosition()[2] };
            // std::vector<double> panel_v = { best_panel_pos[p][0], best_panel_pos[p][1], best_panel_pos[p][2] };
            // e_v.push_back( centroid_v );
            // e_v.push_back( panel_v );
            // double eigenval[3] = { min_dist[p], 0, 0 };
            // larlite::pcaxis llpca( true, 1, eigenval, e_v, centroid_v.data(), 0, itrack );
            // //ev_pcaout->emplace_back( std::move(llpca) );
            
          } // close enough to a hit to register a match
        }//if valid hit found for plane
      }//end of loop over CRT planes
      
    }//end of loop over tracks

    // sort the all rank list
    std::sort( _all_rank_v.begin(), _all_rank_v.end() );

    // sort each hit list
    for ( auto& hitmatches : _hit2track_rank_v ) {
      std::sort( hitmatches.begin(), hitmatches.end() );
    }

    printHitInfo();
    
    return;
  }

  void CRTHitMatch::printHitInfo() {
    
    LARCV_NORMAL() << "===============================================" << std::endl;
    LARCV_NORMAL() << "[ CRT HIT INFO ]" << std::endl;
    for ( size_t idx=0; idx<_crthit_v.size(); idx++ ) {
      auto const& hit = _crthit_v.at(idx);
      std::stringstream ss;
      ss << " [hit " << idx
         << " p=" << hit.plane
         << " pos=(" << hit.x_pos << ", " << hit.y_pos << ", " << hit.z_pos << ") "
         << " t=" << hit.ts2_ns*1.0e-3 << " usec ]";
      
      ss << " matches[";
      for ( auto const& hidx : _hit2track_rank_v[idx] )
        ss << " (" << hidx.dist2hit << "," << hidx.tracklen << ")";
      ss << " ] ";
      LARCV_NORMAL() << ss.str() << std::endl;
    }
    LARCV_NORMAL() << "===============================================" << std::endl;    
    
  }

  float CRTHitMatch::makeOneMatch( const larlite::pcaxis& lfcluster_axis,
                                   const larlite::crthit& hit,
                                   std::vector<float>& panel_pos ) {
    
    int   crt_plane_dim[4] = {        1,        0,       0,      1 }; // Y-axis, X-axis, X-axis, Y-axis
    float endpts[2][3];
    float center[3];
    float dir[3];
    float crtpos[3] = { hit.x_pos, hit.y_pos, hit.z_pos };
    float len = 0;
    for ( int i=0; i<3; i++ ) {
      endpts[0][i] = lfcluster_axis.getEigenVectors()[3][i];
      endpts[1][i] = lfcluster_axis.getEigenVectors()[4][i];
      center[i]    = lfcluster_axis.getAvePosition()[i];
      dir[i] = endpts[0][i]-center[i];
      len += dir[i]*dir[i];
    }
    if (len>0) {
      len = sqrt(len);
      for (int i=0; i<3; i++) dir[i] /= len;
    }

    float max_x = 0.;
    float min_x = 1.0e9;
    for (int i=0; i<2; i++ ) {
      if ( endpts[i][0] > max_x )
        max_x = endpts[i][0];
      if ( endpts[i][0] < min_x )
        min_x = endpts[i][0];
    }
    
    // check if parallel to the plane
    if ( dir[ crt_plane_dim[ hit.plane ] ]==0 ) {
      return -1;
    }

    // check if in time
    float x_offset = hit.ts2_ns*0.001*larutil::LArProperties::GetME()->DriftVelocity();
    if ( max_x-x_offset >= 260.0 || max_x-x_offset<-10.0 ) {
      return -1;
    }
    if ( min_x-x_offset >= 260.0 || min_x-x_offset<-10.0 ) {
      return -1;
    }

    // in time, so let's calculte the position onto the plane
    // remove the time offset
    center[0] -= x_offset;

    // project onto crt plane
    float s = (crtpos[ crt_plane_dim[ hit.plane ] ] - center[ crt_plane_dim[hit.plane] ])/dir[ crt_plane_dim[hit.plane] ];
    panel_pos.resize(3,0);
    for ( int i=0; i<3; i++ ) {
      panel_pos[i] = center[i] + s*dir[i];
    }

    // calculate distance to the hit
    float dist = 0.;
    for (int i=0; i<3; i++) {
      dist += ( crtpos[i]-panel_pos[i] )*( crtpos[i]-panel_pos[i] );
    }
    dist = sqrt( dist );

    panel_pos[0] += x_offset;

    return dist;
  }

  /**
   * get length of pc axis for cluster
   *
   */
  float CRTHitMatch::getLength( const larlite::pcaxis& pca ) {
    float dist = 0.;
    for (int i=0; i<3; i++ ) {
      float dx = ( pca.getEigenVectors()[3][i]-pca.getEigenVectors()[4][i] );
      dist += dx*dx;
    }
    return sqrt(dist);
  }

  /**
   * match opflashes to crttrack_t using closest time
   *
   * ties are broken based on closest centroids based 
   *   on pca-center of tracks and charge-weighted mean of flashes
   * the crttrack_t objects are assumed to have been made in _find_optimal_track(...)
   */
  void CRTHitMatch::_matchOpflashes( const std::vector< const larlite::opflash* >& flash_v,
                                     const std::vector< const larlite::crthit* >&  hit_v,
                                     const std::vector< larlite::larflowcluster >& cluster_v,
                                     std::vector< larlite::opflash >& matched_opflash_v ) {

    std::vector< int > used_flash_v( flash_v.size(), 0 );
    const float _max_dt_flash_crt = 2.0;

    for ( size_t ihit=0; ihit<hit_v.size(); ihit++ ) {

      auto const& crt = *hit_v[ihit];
        
      std::vector< const larlite::opflash* > matched_in_time_v;
      std::vector<float> dt_usec_v;
      std::vector< int > matched_index_v;

      float closest_time = 10e9;

      for ( size_t i=0; i<flash_v.size(); i++ ) {
        if ( used_flash_v[i]>0 ) continue;
          
        float dt_usec = crt.ts2_ns*1.0e-3 - flash_v[i]->Time();

        if ( fabs(dt_usec) < closest_time )
          closest_time = fabs(dt_usec);

        if ( fabs(dt_usec)<_max_dt_flash_crt ) {
          matched_in_time_v.push_back( flash_v[i] );
          dt_usec_v.push_back( dt_usec );
          matched_index_v.push_back( i );
          LARCV_INFO() << "  ...  crt-track and opflash matched. dt_usec=" << dt_usec << std::endl;
        }
          
      }//end of flash loop

      LARCV_NORMAL() << "crt-hit has " << matched_index_v.size() << " flash matches. closest time=" << closest_time  << std::endl;

      
      if ( matched_in_time_v.size()==0 ) {
        // make empty opflash at the time of the crttrack
        std::vector< double > PEperOpDet(32,0.0);
        larlite::opflash blank_flash( crt.ts2_ns*1.0e-3,
                                      0.0,
                                      crt.ts2_ns*1.0e-3,
                                      0,
                                      PEperOpDet );
        matched_opflash_v.emplace_back( std::move(blank_flash) );
      }
      else if ( matched_in_time_v.size()==1 ) {
        // store a copy of the single matched opflash
        matched_opflash_v.push_back( *matched_in_time_v[0] );
        used_flash_v[ matched_index_v[0] ] = 1;
      }
      else {

        // choose closest using distance from center of flash and track
        // need mean of cluster
        std::vector<float> cluster_mean(3,0);
        for ( auto const& hit : cluster_v[ ihit ] ) {
          for (int i=0; i<3; i++)
            cluster_mean[i] += hit[i];
        }
        for (int i=0; i<3; i++ )
          cluster_mean[i] /= (float)cluster_v[ihit].size();
        
        float smallest_dist = 1.0e9;
        const larlite::opflash* closest_opflash = nullptr;

        // we have a loose initial standard. if more than one, we use a tighter standard
        bool have_tight_match = false;
        for ( auto& dt_usec : dt_usec_v ) {
          if ( fabs(dt_usec)<1.5 ) have_tight_match = true;
        }

        for ( size_t i=0; i<matched_in_time_v.size(); i++ ) {

          // ignore loose match if we know we have at least one good one
          if ( have_tight_match && dt_usec_v[i]>1.5 ) continue;

          // get mean of opflash
          std::vector<float> flashcenter = { 0.0,
                                             (float)matched_in_time_v[i]->YCenter(),
                                             (float)matched_in_time_v[i]->ZCenter() };

          float dist = 0.;
          for (int v=1; v<3; v++ ) {
            dist += ( flashcenter[v]-cluster_mean[v] )*( flashcenter[v]-cluster_mean[v] );
          }
          dist = sqrt(dist);

          if ( dist<smallest_dist ) {
            smallest_dist = dist;
            closest_opflash = matched_in_time_v[i];
          }
          LARCV_INFO() << "  ... distance between opflash and track center: " << dist << " cm" << std::endl;
        }//end of candidate opflash match
        
        // store best match
        if ( closest_opflash ) {
          matched_opflash_v.push_back( *closest_opflash );
        }
        else {
          // shouldnt get here
          throw std::runtime_error( "should not get here" );
        }
      }//else more than 1 match
          
    }//end of CRT HIT loop

    // check we have the right amount of flashes
    if ( matched_opflash_v.size()!=cluster_v.size() || matched_opflash_v.size()!=hit_v.size() ) {
      throw std::runtime_error( "different amount of input crt-hit, cluster, and opflashes");
    }
    
  }

  /**
   * using sorted match list per hit, we determine if we should merge the clusters
   *
   */
  larlite::larflowcluster CRTHitMatch::_merge_matched_cluster( const std::vector< CRTHitMatch::match_t >& hit_match_v,
                                                               std::vector<int>& used_in_merge,
                                                               bool& merged ) {

    merged = false;
    larlite::larflowcluster lfcluster;

    // if one or zero matches, nothing to do here.
    if ( hit_match_v.size()<=1 )
      return lfcluster;

    // we use cluster tools, so start by converting best track 
    larflow::reco::cluster_t mergecluster;
    int nmerged = 0;
    for ( size_t idx=0; idx<hit_match_v.size(); idx++ ) {
      LARCV_INFO() << "merger: make cluster for larflowcluster index=" << hit_match_v[idx].trackidx << " nhits=" << _lfcluster_v[ hit_match_v[idx].trackidx ]->size() << std::endl;
      larflow::reco::cluster_t c = larflow::reco::cluster_from_larflowcluster( *_lfcluster_v[ hit_match_v[idx].trackidx ] );
      if ( nmerged==0 ) {
        // first cluster
        LARCV_INFO() << "merger: set first cluster" << std::endl;
        std::swap(mergecluster,c);
        used_in_merge[idx] = 1;
        merged = true;
        nmerged++;
        continue;
      }

      // determine a good match
      LARCV_INFO() << "merger: test for merger" << std::endl;      
      std::vector< std::vector<float> > closestpts_vv;
      float endptdist = larflow::reco::cluster_closest_endpt_dist( mergecluster, c, closestpts_vv );
      float cospca = cluster_cospca( mergecluster, c );
      cospca = 1.0 - fabs(cospca);

      if ( float(cospca)*180.0/3.14159 < 20.0 && endptdist < 30.0 ) {
        LARCV_INFO() << "merge test passed" << std::endl;      
        larflow::reco::cluster_t testmerge = larflow::reco::cluster_merge( mergecluster, c );

        if ( testmerge.pca_eigenvalues[1] < 30.0 ) {
          merged = true;
          used_in_merge[idx] = 1;
          nmerged++;
          LARCV_INFO() << "swap for merged cluster" << std::endl;                
          std::swap( mergecluster, testmerge );
        }
        
      }
    }

    for ( size_t idx=0; idx<hit_match_v.size(); idx++ ) {
      if ( used_in_merge[idx]==1 ) {
        for ( auto const& hit : *_lfcluster_v[ hit_match_v[idx].trackidx ] )
          lfcluster.push_back( hit );
      }
    }

    return lfcluster;
  }

  /**
   * save products to storage_manager
   *
   */
  void CRTHitMatch::save_to_file( larlite::storage_manager& outll, bool remove_if_no_flash ) {

    // now store data
    // --------------
    // (1) new crt object with updated hit positions (and old hit positions as well)
    // (2) matched opflash objects
    // (3) larflow3dhit clusters which store 3d pos and corresponding imgcoord locations for each track
    // larlite::event_crthit* out_crthit
    //   = (larlite::event_crthit*)outll.get_data( larlite::data::kCRTHit, "matchcrthit" );
    larlite::event_crttrack* out_crttrack
      = (larlite::event_crttrack*)outll.get_data( larlite::data::kCRTTrack, "matchcrthit" );
    larlite::event_opflash* out_opflash
      = (larlite::event_opflash*)outll.get_data( larlite::data::kOpFlash, "matchcrthit" );
    larlite::event_larflowcluster* out_lfcluster
      = (larlite::event_larflowcluster*)outll.get_data( larlite::data::kLArFlowCluster, "matchcrthit" );

    for (size_t i=0; i<_matched_cluster.size(); i++ ) {
      auto& cluster = _matched_cluster[i];
      auto& opflash = _matched_opflash_v[i];
      auto const& crthit  = *_match_hit_v[i];
      auto const& clustert = _matched_cluster_t[i];
      
      if ( remove_if_no_flash && opflash.TotalPE()==0.0 ) {
        LARCV_INFO() << "no matching flash for matched CRT hit[" << i << "], not saving" << std::endl;
        continue;
      }
      
      float petot = opflash.TotalPE();
      LARCV_NORMAL() << "saving track with opflash pe=" << petot << " nopdets=" << opflash.nOpDets() << std::endl;

      // form a crt-track
      larlite::crttrack outtrack;
      // outtrack.feb_id = crthit.feb_id;
      // outtrack.pesmap = crthit.pesmap;
      // outtrack.peshit = crthit.peshit;
      // outtrack.ts0_s       = crthit.ts0_s;
      // outtrack.ts0_ns      = crthit.ts0_ns;
      // outtrack.ts0_ns_corr = crthit.ts0_ns_corr;

      // first crt hit
      outtrack.ts2_ns_h1  = crthit.ts2_ns;
      outtrack.plane1     = crthit.plane;
      outtrack.x1_pos     = crthit.x_pos;
      outtrack.y1_pos     = crthit.y_pos;
      outtrack.z1_pos     = crthit.z_pos;

      // second crt hit: use the farthest point on the cluster
      outtrack.plane2     = -1;
      outtrack.ts2_ns_h2 = crthit.ts2_ns;

      std::vector<float> crtpos = { outtrack.x1_pos, outtrack.y1_pos, outtrack.z1_pos };

      int farend = 0;
      float fardist = 0.;
      for ( int iend=0; iend<2; iend++ ) {
        float dist = 0;
        for (int i=0; i<3; i++ ) {
          dist += ( crtpos[i]- clustert.pca_ends_v[iend][i] )*( crtpos[i]- clustert.pca_ends_v[iend][i] );
        }
        dist = sqrt(dist);
        if ( dist>fardist ) {
          fardist = dist;
          farend = iend;
        }
      }

      outtrack.x2_pos = clustert.pca_ends_v[farend][0];
      outtrack.y2_pos = clustert.pca_ends_v[farend][1];
      outtrack.z2_pos = clustert.pca_ends_v[farend][2];      
      
      out_crttrack->push_back(outtrack);
      out_opflash->push_back(opflash);
      out_lfcluster->push_back( cluster );
      
    }
    
  }
  
}
}
