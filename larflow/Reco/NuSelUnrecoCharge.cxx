#include "NuSelUnrecoCharge.h"

#include <sstream>

#include "larcv/core/DataFormat/EventImage2D.h"
#include "larcv/core/DataFormat/EventPixel2D.h"

#include "ClusterImageMask.h"

namespace larflow {
namespace reco {

  void NuSelUnrecoCharge::analyze( larcv::IOManager& iolcv,
                                   larlite::storage_manager& ioll,
                                   larflow::reco::NuVertexCandidate& nuvtx,
                                   larflow::reco::NuSelectionVariables& output )
  {

    // what are the measures?
    // (1) define nearby clusters:  clusters with hits some distance
    //     from the vertex, track end-points.
    // (2) can count number of hits in these clusters.
    // (3) can count total charge in these clusters on the three planes.

    // we do we get the clusters?
    // we've also lost track of where our clusters came from?

    // first thing to do is make a mask of where our charge is.
    const float adc_threshold = 10;
    
    std::vector< std::string > cluster_producers
      = { "trackprojsplit_wcfilter", "showergoodhit" };

    larcv::EventImage2D* ev_img
      = (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D, "wire" );
    auto const& adc_v = ev_img->as_vector();

    // get image2d with pixels from nu candidate clusters
    ClusterImageMask masker;
    std::vector<larcv::Image2D> nuvtx_mask_v = masker.makeChargeMask( nuvtx, adc_v );

    LARCV_DEBUG() << "Number of pixels masked by nu candidate: " << masker._npix << std::endl;

    // calculate metric(s) to cut
    // how many of the untagged pixels are used?
    larcv::EventImage2D* ev_thrumu
      = (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D, "thrumu");
    auto const& thrumu_v =  ev_thrumu->as_vector();
    std::vector<int> unreco_counts(adc_v.size(),0);
    std::vector<float> unreco_fraction(adc_v.size(),0);
    _count_unreco_pixels( nuvtx_mask_v, adc_v, thrumu_v, adc_threshold,
                          unreco_counts, unreco_fraction );

    // get/define mask container
    larcv::EventImage2D* evout_mask
      = (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D, "nuvtx_mask" );
    larcv::EventPixel2D* evout_pix
      = (larcv::EventPixel2D*)iolcv.get_data(larcv::kProductPixel2D, "nuvtx_mask" );
    
    if ( _ksave_mask ) {
      
      for ( auto& img : nuvtx_mask_v ) {

        // make pixel cluster
        larcv::Pixel2DCluster pixcluster;
        for (int c=0; c<(int)img.meta().cols(); c++) {
          for (int r=0; r<(int)img.meta().rows(); r++) {
            if ( img.pixel(r,c)>0 ) {
              larcv::Pixel2D pix(c,r);
              pixcluster += pix;
            }
          }
        }
        evout_pix->Append( img.meta().plane(), pixcluster );
        
        evout_mask->Emplace( std::move(img) );
      }
    }
    
  }
                                   
  void NuSelUnrecoCharge::_count_unreco_pixels( std::vector<larcv::Image2D>& numask_v,
                                                const std::vector<larcv::Image2D>& adc_v,
                                                const std::vector<larcv::Image2D>& thrumu_v,
                                                const float adc_threshold,
                                                std::vector<int>& unreco_counts,
                                                std::vector<float>& unreco_fraction )
  {

    unreco_counts.resize( adc_v.size(), 0 );
    unreco_fraction.resize( adc_v.size(), 0 );
    std::vector<int> unreco_intime_counts(adc_v.size(), 0);
    clearVars();
    
    for (int p=0; p<(int)adc_v.size(); p++) {
      unreco_counts[p] =  0;
      unreco_fraction[p] = 0.;

      auto const& img = adc_v[p];
      auto & mask = numask_v[p];
      auto const& thrumu = thrumu_v[p];
      auto const& meta = adc_v[p].meta();
      if ( meta.rows()!=mask.meta().rows() || meta.cols()!=mask.meta().cols() ) {
        LARCV_CRITICAL() << "dimensions of adc image and nu-candidate mask image are not the same" << std::endl;
      }
      if ( meta.rows()!=thrumu.meta().rows() || meta.cols()!=thrumu.meta().cols() ) {
        LARCV_CRITICAL() << "dimensions of adc image and cosmic-tagged image are not the same" << std::endl;
      }

      
      for (int r=0; r<(int)meta.rows(); r++) {
        for (int c=0; c<(int)meta.cols(); c++) {
          float imgval  = img.pixel(r,c,__FILE__,__LINE__);
          float maskval = mask.pixel(r,c,__FILE__,__LINE__);
          float tagval  = thrumu.pixel(r,c,__FILE__,__LINE__);

          if ( imgval>=adc_threshold && tagval==0 ) {
            // pixel with content and not cosmic-tagged
            unreco_intime_counts[p]++;

            if ( maskval==0 ) {
              unreco_counts[p]++;
              unreco_fraction[p]++;
              mask.set_pixel(r,c,2.0); /// for debug
            }
          }
               
        }
      }

      if ( unreco_intime_counts[p]>0 ) {
        unreco_fraction[p] /= (float)unreco_intime_counts[p];
      }
      
    }//end of plane loop


    if ( _tree ) {
      // set tree vars
      _intime_count_v = unreco_intime_counts;
      _unreco_count_v = unreco_counts;
      _unreco_fraction_v = unreco_fraction;

      std::vector<float> copy_frac = _unreco_fraction_v;
      std::sort( copy_frac.begin(), copy_frac.end() );
      _min_fraction = copy_frac.front();
      _max_fraction = copy_frac.back();
      if ( copy_frac.size()>0 ) {
        if ( copy_frac.size()%2==0 ) {
          // even
          int n = (int)copy_frac.size()/2;
          _median_fraction = 0.5*( copy_frac[n-1] + copy_frac[n] );
        }
        else {
          // odd
          int n = (int)copy_frac.size()/2;
          _median_fraction = copy_frac[n];
        }
      }
    }
    
    LARCV_INFO() << "Results" << std::endl;
    std::stringstream ss_intime;
    ss_intime << "  intime counts: ";
    for (auto const& count : unreco_intime_counts )
      ss_intime << count << " ";
    std::stringstream ss_unreco;
    ss_unreco << "  unreco counts: ";
    for (auto const& count : unreco_counts )
      ss_unreco << count << " ";
    std::stringstream ss_frac;
    for (auto const& frac : unreco_fraction )
      ss_frac << frac << " ";
    
    LARCV_INFO() << ss_intime.str() << std::endl;
    LARCV_INFO() << ss_unreco.str() << std::endl;
    LARCV_INFO() << ss_frac.str() << std::endl;
    
  }

  void NuSelUnrecoCharge::clearVars()
  {
    _intime_count_v.clear();
    _unreco_count_v.clear();
    _unreco_fraction_v.clear();
    _median_fraction = 0;
    _min_fraction = 0;
    _max_fraction = 0;
  }

  void NuSelUnrecoCharge::bindVarsToTree( TTree* tree )
  {
    if ( _tree ) {
      LARCV_CRITICAL() << "Tree already bound!" << std::endl;
    }

    _tree = tree;
    _tree->Branch( "nusel_unrecoq_intime_v", &_intime_count_v );
    _tree->Branch( "nusel_unrecoq_count_v", &_intime_count_v );
    _tree->Branch( "nusel_unrecoq_fraction_v", &_intime_count_v );
    _tree->Branch( "nusel_unrecoq_median", &_median_fraction, "nusel_unrecoq_median/F" );
    _tree->Branch( "nusel_unrecoq_min", &_min_fraction, "nusel_unrecoq_min/F" );
    _tree->Branch( "nusel_unrecoq_max", &_max_fraction, "nusel_unrecoq_max/F" );
    
  }
  
}
}
