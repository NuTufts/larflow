#include "TruthRecoMatching.h"

namespace larflow {
namespace reco {

  TruthRecoMatching::TruthRecoMatching()
    : larcv::larcv_base("TruthRecoMatching")
  {}

  TruthRecoMatching::PixelMatch_t
  TruthRecoMatching::calculate_pixel_completeness( const std::vector<larcv::Image2D>& wire_v,
                                                   const std::vector<larcv::Image2D>& instance_v,
                                                   const larflow::reco::NuVertexCandidate& nuvtx,
                                                   const float adc_threshold )
  {

    size_t nplanes = wire_v.size();
    
    // first we project reco clusters into an image
    std::vector<larcv::Image2D> proj_v;
    for (size_t p=0; p<wire_v.size(); p++) {      
      larcv::Image2D projection(wire_v[p].meta());
      projection.paint(0);
      proj_v.emplace_back( std::move(projection) );
    }

    // project tracks and showers into image
    for (int itrack=0; itrack<(int)nuvtx.track_v.size(); itrack++) {
      auto const& trackcluster = nuvtx.track_hitcluster_v.at(itrack);
      for (auto const& hit : trackcluster ) {
        for (size_t p=0; p<nplanes; p++) {
          int wire = hit.targetwire[p];
          int tick = hit.tick;

          if ( tick<wire_v[p].meta().min_y() || tick>=wire_v[p].meta().max_y()
               || wire<wire_v[p].meta().min_x() || wire>=wire_v[p].meta().max_x() ) {
            continue;
          }

          int row = wire_v[p].meta().row(tick,__FILE__,__LINE__);
          int col = wire_v[p].meta().col(wire,__FILE__,__LINE__);

          float adcval = 0.;
          proj_v[p].set_pixel(row,col,1.0);
        }
      }
    }
    for (int ishower=0; ishower<(int)nuvtx.shower_v.size(); ishower++) {
      auto const& showercluster = nuvtx.shower_v.at(ishower);
      for (auto const& hit : showercluster ) {
        for (size_t p=0; p<nplanes; p++) {
          int wire = hit.targetwire[p];
          int tick = hit.tick;

          if ( tick<wire_v[p].meta().min_y() || tick>=wire_v[p].meta().max_y()
               || wire<wire_v[p].meta().min_x() || wire>=wire_v[p].meta().max_x() ) {
            continue;
          }

          int row = wire_v[p].meta().row(tick,__FILE__,__LINE__);
          int col = wire_v[p].meta().col(wire,__FILE__,__LINE__);

          float adcval = 0.;
          proj_v[p].set_pixel(row,col,1.0);
        }
      }
    }


    PixelMatch_t result;
    
    std::vector<float> true_visible_charge(nplanes,0);
    std::vector<float> reco_visible_charge(nplanes,0);
    std::vector<float> misreco_visible_charge(nplanes,0);    
    std::vector<float> frac_visible_charge(nplanes,0);

    std::vector<int> true_visible_pixels(nplanes,0);
    std::vector<int> reco_visible_pixels(nplanes,0);
    std::vector<int> misreco_visible_pixels(nplanes,0);    
    std::vector<float> frac_visible_pixels(nplanes,0);
    
    for (size_t p=0;p<3;p++) {
      auto const& instance = instance_v[p].as_vector();
      auto const& adc = wire_v[p].as_vector();
      auto const& proj = proj_v[p].as_vector();

      for (int i=0;i<(int)adc.size();i++) {
        if ( instance[i]>0 && adc[i]>adc_threshold ) {
          true_visible_charge[p] += adc[i];
          true_visible_pixels[p]++;
          if ( proj[i]>0 ) {
            reco_visible_charge[p] += adc[i];
            reco_visible_pixels[p]++;
          }
        }
        if ( instance[i]<=0 && adc[i]>adc_threshold && proj[i]>0 ) {
          misreco_visible_charge[p] += adc[i];
          misreco_visible_pixels[p]++;
        }
      }
    }

    result.true_visible_charge=true_visible_charge;
    result.reco_visible_charge=reco_visible_charge;
    result.misreco_visible_charge=misreco_visible_charge;
    result.frac_visible_charge=frac_visible_charge;
    result.true_visible_pixels=true_visible_pixels;
    result.reco_visible_pixels=reco_visible_pixels;
    result.misreco_visible_pixels=misreco_visible_pixels;
    result.frac_visible_pixels=frac_visible_pixels;
    
  }

  
}
}
