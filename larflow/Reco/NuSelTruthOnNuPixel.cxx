#include "NuSelTruthOnNuPixel.h"

#include "larcv/core/DataFormat/EventImage2D.h"

namespace larflow {
namespace reco {

  /**
   * @brief count fraction of true nu pixels around reco vertex
   *
   */
  void NuSelTruthOnNuPixel::analyze( larcv::IOManager& iolcv,
                                     larlite::storage_manager& ioll,
                                     larflow::reco::NuVertexCandidate& nuvtx,
                                     larflow::reco::NuSelectionVariables& output )
  {

    larcv::EventImage2D* ev_adc
      = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, "wire" );
    auto const& adc_v = ev_adc->as_vector();
    larcv::EventImage2D* ev_segment
      = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, "segment" );
    auto const& seg_v = ev_segment->as_vector();

    int dpix=3;
    std::vector<int> nabove(adc_v.size(),0);
    std::vector<int> nnu(adc_v.size(),0);

    auto const& meta = adc_v.front().meta();
    
    for (int r=-dpix; r<=dpix; r++) {

      int row = nuvtx.row + r;
      if ( row<0 || row>=(int)meta.rows() )
        continue;

      for (int p=0; p<(int)adc_v.size(); p++) {
        auto const& adc = adc_v[p];
        auto const& seg = seg_v[p];
        
        for (int c=-dpix;c<=dpix;c++) {
          int col = nuvtx.col_v[p] + c;

          if ( col<0 || col>=(int)meta.cols() )
            continue;
          
          float pixval = adc.pixel(row,col,__FILE__,__LINE__);
          float segval = seg.pixel(row,col,__FILE__,__LINE__);

          if ( pixval>10.0 ) {
            nabove[p]++;
            if ( segval>0 )
              nnu[p]++;
          }
        }//end of col loop
      }//end of plane loop
    }//end of row loop

    float tot_nabove = 0.;
    float tot_nnu    = 0.;
    for ( int p=0; p<(int)adc_v.size(); p++ ) {
      tot_nabove += (float)nabove[p];
      tot_nnu    += (float)nnu[p];
    }

    if ( tot_nabove>0 ) {
      output.truth_vtxFracNu = tot_nnu/tot_nabove;
    }
    else {
      output.truth_vtxFracNu = 0.;
    }
      
  }

}
}
