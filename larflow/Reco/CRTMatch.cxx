#include "CRTMatch.h"

#include "LArUtil/LArProperties.h"

namespace larflow {
namespace reco {

  void CRTMatch::clear() {
    _intime_opflash_v.clear();
    _outtime_opflash_v.clear();
    _crthit_v.clear();
    _crttrack_v.clear();
    _lfcluster_v.clear();
    _pcaxis_v.clear();
  }
  
  void CRTMatch::addIntimeOpFlashes( const larlite::event_opflash& opflash_v ) {
    for ( auto const& opf : opflash_v )
      _intime_opflash_v.push_back( opf );
  }

  void CRTMatch::addCosmicOpFlashes( const larlite::event_opflash& opflash_v ) {
    for ( auto const& opf : opflash_v )
      _outtime_opflash_v.push_back( opf );
  }

  void CRTMatch::addCRThits( const larlite::event_crthit& crthit_v ) {
    for ( auto const& hit : crthit_v )
      _crthit_v.push_back( hit );
  }
  
  void CRTMatch::addCRTtracks( const larlite::event_crttrack& crttrack_v ) {
    for ( auto const& track : _crttrack_v )
      _crttrack_v.push_back( track );
  }
  
  void CRTMatch::addLArFlowClusters( const larlite::event_larflowcluster& lfcluster_v,
                                     const larlite::event_pcaxis& pcaxis_v ) {
    for ( auto const& cluster : lfcluster_v ) {
      _lfcluster_v.push_back( &cluster );
    }
    
    for ( auto const& pca : pcaxis_v ) {
      _pcaxis_v.push_back( &pca );
    }
  }
  

  void CRTMatch::match( larlite::storage_manager& llio, larlite::storage_manager& outio ) {

    _hit2trackidx_v.resize( _crthit_v.size() );
    printHitInfo();

    float crt_plane_pos[4] = { -261.606, -142.484, 393.016, 618.25 }; // bottom, side, side, top

    larlite::event_pcaxis* ev_pcaout = (larlite::event_pcaxis*)outio.get_data( larlite::data::kPCAxis, "crtmatch" );

    std::cout << "==============================" << std::endl;
    std::cout << "[ CRTMatch ]" << std::endl;
    for (size_t itrack=0; itrack<_pcaxis_v.size(); itrack++ ) {
      auto const& pca = *_pcaxis_v[itrack];

      float len = getLength(pca);
      if ( len<10.0 ) continue;

      // allocate vars for best intersection for each CRT plane
      float min_dist[4] = { 1.0e9, 1.0e9, 1.0e9, 1.0e9 };
      int best_hitidx[4] = {-1, -1, -1, -1};
      std::vector<float> best_panel_pos[4];
      for (int i=0; i<4; i++) best_panel_pos[i].resize(3,0);
      
      for (size_t jhit=0; jhit<_crthit_v.size(); jhit++) {
        auto const& crthit = _crthit_v[jhit];
        int crtplane = crthit.plane;

        std::vector<float> panel_pos;
        float dist = makeOneMatch( pca, crthit, panel_pos );

        if ( dist>0 && dist<min_dist[crtplane] ) {
          min_dist[crtplane] = dist;
          best_hitidx[crtplane] = jhit;
          best_panel_pos[crtplane] = panel_pos;
        }
      }//end of hit loop
      std::cout << " [" << itrack << "] closest dist per plane = "
                << "[ " << min_dist[0] << ", " << min_dist[1] << ", " << min_dist[2] << ", " << min_dist[3] << "]"
                << std::endl;

      for (int p=0; p<4; p++ ) {
        if ( best_hitidx[p]>=0 ) {
          auto const& besthit = _crthit_v[best_hitidx[p]];        
          std::cout << " crt_hit=(" << besthit.x_pos << ", " << besthit.y_pos << "," << besthit.z_pos << ") ";
          std::cout << " panel_pos=(" << best_panel_pos[p][0] << "," << best_panel_pos[p][1] << "," << best_panel_pos[p][2] << ") ";
          std::cout << std::endl;
          _hit2trackidx_v[ best_hitidx[p] ].push_back( itrack );

          if ( min_dist[p]>=0 && min_dist[p]<30.0 ) {
            float line[3] = {0};
            float len = 0.;
            for (int i=0; i<3; i++ ) {
              line[i] = best_panel_pos[p][i]-pca.getAvePosition()[i];
              len += line[i]*line[i];
            }
            len = sqrt(len);
            for (int i=0; i<3; i++ ) line[i] /= len;
          
            // store for visualization purposes
            larlite::pcaxis::EigenVectors e_v; // 3-axis + 2-endpoints
            for ( size_t p=0; p<3; p++ ) {
              std::vector<double> da_v = { line[0], line[1], line[2] };
              e_v.push_back( da_v );
            }
            std::vector<double> centroid_v = { pca.getAvePosition()[0], pca.getAvePosition()[1], pca.getAvePosition()[2] };
            std::vector<double> panel_v = { best_panel_pos[p][0], best_panel_pos[p][1], best_panel_pos[p][2] };
            e_v.push_back( centroid_v );
            e_v.push_back( panel_v );
            double eigenval[3] = { min_dist[p], 0, 0 };
            larlite::pcaxis llpca( true, 1, eigenval, e_v, centroid_v.data(), 0, itrack );
            ev_pcaout->emplace_back( std::move(llpca) );
          }
        }
      }//end of loop over CRT planes
      
    }

    printHitInfo();
    
    return;
  }

  void CRTMatch::printHitInfo() {
    
    std::cout << "===============================================" << std::endl;
    std::cout << "[ CRT HIT INFO ]" << std::endl;
    for ( size_t idx=0; idx<_crthit_v.size(); idx++ ) {
      auto const& hit = _crthit_v.at(idx);
      std::cout << " [" << idx << "] ";
      
      std::cout << " matches[";
      for ( auto const& hidx : _hit2trackidx_v[idx] )
        std::cout << " " << hidx;
      std::cout << " ] ";
      
      std::cout << "(p=" << hit.plane
                << ", " << hit.x_pos << ", " << hit.y_pos << ", " << hit.z_pos << ") "
                << "t=" << hit.ts2_ns*1.0e-3 << " usec"
                << std::endl;
    }
    std::cout << "===============================================" << std::endl;    
    
  }

  float CRTMatch::makeOneMatch( const larlite::pcaxis& lfcluster_axis, const larlite::crthit& hit,
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

  float CRTMatch::getLength( const larlite::pcaxis& pca ) {
    float dist = 0.;
    for (int i=0; i<3; i++ ) {
      float dx = ( pca.getEigenVectors()[3][i]-pca.getEigenVectors()[4][i] );
      dist += dx*dx;
    }
    return sqrt(dist);
  }
  
}
}
