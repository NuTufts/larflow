#include "NuSelVertexVars.h"

#include "larcv/core/DataFormat/EventImage2D.h"
#include "DataFormat/larflow3dhit.h"

namespace larflow {
namespace reco {

  /**
   * @brief analyze charge and ssnet scores around vertex
   * 
   */
  void NuSelVertexVars::analyze( larcv::IOManager& iolcv,
                                 larlite::storage_manager& ioll,
                                 larflow::reco::NuVertexCandidate& nuvtx,
                                 larflow::reco::NuSelectionVariables& output )
  {

    // pixel values
    larcv::EventImage2D* ev_img
      = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, "wire" );
    auto const& adc_v = ev_img->as_vector();

    // five-particle ssnet
    larcv::EventImage2D* ev_ssnet
      = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, "fivepidssn" );
    auto const& ssnet_v = ev_ssnet->as_vector();

    // get original keypoint hit
    larlite::event_larflow3dhit* ev_keypoint
      = (larlite::event_larflow3dhit*)ioll.get_data( larlite::data::kLArFlow3DHit, nuvtx.keypoint_producer );
    auto const& keypoint_hit = ev_keypoint->at(nuvtx.keypoint_index);

    std::vector<float> charge_pixsum( adc_v.size(), 0 );
    std::vector<int>   nabove_thresh( adc_v.size(), 0 );
    std::vector< std::vector<int> > particle_sum( adc_v.size() );
    for (int p=0; p<(int)adc_v.size(); p++)
      particle_sum[p].resize(5,0);

    int row = adc_v.front().meta().row( (float)nuvtx.tick, __FILE__, __LINE__ );    
    LARCV_INFO() << "vertex imgcoord("
                 << nuvtx.col_v[0] << ","
                 << nuvtx.col_v[1] << ","
                 << nuvtx.col_v[2] << ") "
                 << " row=" << row
                 << " tick=" << nuvtx.tick
                 << std::endl;
    LARCV_INFO() << "keypoint type: " << (int)keypoint_hit[3]
                 << " imgcoord: ("
                 << keypoint_hit.targetwire[0] << ","
                 << keypoint_hit.targetwire[1] << ","
                 << keypoint_hit.targetwire[2] << ") "
                 << " tick=" << keypoint_hit.tick
                 << std::endl;
    
    int dpix=2;

    for (int dr=-dpix; dr<=dpix; dr++) {
      int r = row+dr;
      if ( r<0 || r>=(int)adc_v.front().meta().rows() )
        continue;

      for (int p=0; p<(int)adc_v.size(); p++) {
        auto const& img = adc_v[p];
        for(int dc=-dpix; dc<=dpix; dc++) {
          int c = nuvtx.col_v[p] + dc;
          if ( c<0 || c>=(int)img.meta().cols() )
            continue;
          
          float pixval = adc_v[p].pixel( r, c );
          if ( pixval>10 ) {
            nabove_thresh[p]++;          
            int maxpid   = (int)ssnet_v[p].pixel( r, c );
            particle_sum[p][maxpid-1]++; // pid start at 1, so remove 1.
            charge_pixsum[p] += pixval;
          }
        }//dcol
      }//nplanes
    }//drow

    int tot_above_thresh = 0;
    int tot_nproton = 0;
    float totcharge_sum = 0.;
    for (int p=0; p<(int)nabove_thresh.size(); p++) {
      tot_above_thresh += nabove_thresh[p];
      tot_nproton += particle_sum[p][0];
      totcharge_sum += charge_pixsum[p];
    }
    
    if ( tot_above_thresh>0 ) {
      output.vertex_hip_fraction = (float)tot_nproton/(float)tot_above_thresh;
      output.vertex_charge_per_pixel = totcharge_sum/(float)tot_above_thresh;
    }
    else {
      output.vertex_hip_fraction = 0.;
      output.vertex_charge_per_pixel =  0.;
    }
    output.nabove_threshold_vertex_pix_v = nabove_thresh;
    output.vertex_plane_charge_v = charge_pixsum;
    output.vertex_type = (int)keypoint_hit[3];

  }
  
}
}
