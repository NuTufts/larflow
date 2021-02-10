#include "ShortProtonClusterReco.h"

#include "DataFormat/larflowcluster.h"
#include "DataFormat/pcaxis.h"
#include "DataFormat/track.h"
#include "larcv/core/DataFormat/EventImage2D.h"
#include "cluster_functions.h"

namespace larflow {
namespace reco {

  void ShortProtonClusterReco::process( larcv::IOManager& iolcv,
                                        larlite::storage_manager& ioll )
  {

    larlite::event_larflow3dhit* ev_hit
      = (larlite::event_larflow3dhit*)ioll.get_data( larlite::data::kLArFlow3DHit,
                                                     _input_hit_treename );

    larcv::EventImage2D* ev_adc
      = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, "wire" );
                                              
    auto const& meta = ev_adc->at(0).meta();
    
    std::vector< size_t > orig_idx_v;
    std::vector< std::vector<float> > points_v;
    orig_idx_v.reserve( ev_hit->size() );
    points_v.reserve( ev_hit->size() );

    const float hip_threshold[3] = { 55.0, 70.0, 70.0 };
    const float max_length = 20.0;

    larlite::event_larflow3dhit* ev_out_hit 
      = (larlite::event_larflow3dhit*)ioll.get_data( larlite::data::kLArFlow3DHit, "hip" );

    int dpix=0;
    
    for ( size_t ihit=0; ihit<ev_hit->size(); ihit++ ) {
      
      auto const& lfhit = (*ev_hit)[ihit];

      if ( lfhit.tick <= meta.min_y() || lfhit.tick>=meta.max_y() )
        continue;

      int row = meta.row( lfhit.tick, __FILE__, __LINE__ );

      // check to see if it lands on HIP pixel      
      // 2 planes must see above thresh
      bool plane_has_hip[3] = { false, false, false };
      for (int dr=-dpix; dr<=dpix; dr++) {
        int r = row+dr;
        if ( r<0 || r>=(int)meta.rows() )
          continue;
        
        for (int p=0; p<3; p++) {

          if ( plane_has_hip[p] )
            break;
          
          auto const& img = ev_adc->as_vector()[p];

          int col = img.meta().col( lfhit.targetwire[p], __FILE__, __LINE__ );

          for (int dc=-dpix; dc<=dpix; dc++) {
            if ( plane_has_hip[p] )
              break;
            int c = col+dc;
            float pixval = img.pixel( r, c );
            if ( pixval>hip_threshold[p] ) {
              plane_has_hip[p] = true;
            }
          }
        }
      }
      int nplanes_above_thresh = 0;
      for (int p=0; p<3; p++) {{
          if ( plane_has_hip[p] )
            nplanes_above_thresh++;
        }
      }
      
      if ( nplanes_above_thresh<2 )
        continue;
      
      std::vector<float> pt = { lfhit[0], lfhit[1], lfhit[2] };
      points_v.emplace_back( std::move(pt) );
      orig_idx_v.push_back( ihit );
      ev_out_hit->push_back( lfhit );
      
    }
    LARCV_INFO() << "number of HIP spacepoints: " << points_v.size() << " of " << ev_hit->size() << std::endl;

    
    // perform dbscan
    const float maxdist = 1.0;
    const int minsize = 5;
    const int maxkd = 50;
    std::vector< larflow::reco::cluster_t > cluster_v;
    larflow::reco::cluster_sdbscan_spacepoints( points_v, cluster_v, maxdist, minsize, maxkd );
    larflow::reco::cluster_runpca( cluster_v );

    LARCV_INFO() << "number of clusters: " << cluster_v.size() << std::endl;

    // find short, straight clusters
    std::vector< larflow::reco::cluster_t > proton_candidates_v;
    for ( auto& c : cluster_v ) {
      if ( c.pca_len>max_length )
        continue;

      if ( std::isnan(c.pca_ends_v[0][0]) || std::isnan(c.pca_ends_v[1][0]) || std::isnan(c.pca_len) )
        continue;

      // to do: cut on width somehow?
      proton_candidates_v.emplace_back( std::move(c) );
    }

    // no need for other clusters
    cluster_v.clear();

    // save larflowcluster and pcaxis
    larlite::event_larflowcluster* ev_out_cluster
      = (larlite::event_larflowcluster*)ioll.get_data(larlite::data::kLArFlowCluster,"hip");
    larlite::event_pcaxis* ev_out_pca
      = (larlite::event_pcaxis*)ioll.get_data(larlite::data::kPCAxis,"hip");
    larlite::event_track* ev_out_track
      = (larlite::event_track*)ioll.get_data(larlite::data::kTrack,"hip");

    for (size_t c=0; c<proton_candidates_v.size(); c++) {
      auto const& cluster = proton_candidates_v[c];

      larlite::larflowcluster lfcluster;
      for ( auto& idx : cluster.hitidx_v ) {
        lfcluster.push_back( (*ev_hit)[ orig_idx_v[idx] ] );
      }
      larlite::pcaxis pc = larflow::reco::cluster_make_pcaxis( cluster, c );
      larlite::track lltrack;
      lltrack.reserve(2);

      TVector3 cdir;
      TVector3 cstart;
      TVector3 cend;
      float len = 0.;
      for (int i=0; i<3; i++) {
        cdir[i]   = cluster.pca_axis_v[0][i];
        len += cdir[i]*cdir[i];
        cstart[i] = cluster.pca_ends_v[0][i];
        cend[i]   = cluster.pca_ends_v[1][i];
      }
      len = sqrt(len);
      if ( len>0 ) {
        for (int i=0; i<3; i++)
          cdir[i] /= len;
      }
      lltrack.add_vertex( cstart );
      lltrack.add_direction( cdir );
      lltrack.add_vertex( cend );
      lltrack.add_direction( cdir );

      ev_out_cluster->emplace_back( std::move(lfcluster) );
      ev_out_pca->emplace_back( std::move(pc) );
      ev_out_track->emplace_back( std::move(lltrack) );
    }
    
  }
  
  
}
}
