#include "NuSelUnrecoCharge.h"

#include <sstream>

#include "larcv/core/DataFormat/EventImage2D.h"
#include "larcv/core/DataFormat/EventPixel2D.h"
#include "larlite/DataFormat/larflow3dhit.h"
#include "larlite/DataFormat/larflowcluster.h"


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
      = { "trackprojsplit_wcfilter", "showerkp", "showergoodhit" };

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
    std::vector<int> all_intime_counts;   // num above thresh pixels intime
    std::vector<int> reco_intime_counts;  // reco pixel intime
    std::vector<int> reco_outtime_counts; // reco pixel outtime
    std::vector<int> intime_unreco_counts; // intime but not recod						
    std::vector<float> unreco_fraction;
    std::vector<float> cosmic_reco_fraction;
    _count_unreco_pixels( nuvtx_mask_v, adc_v, thrumu_v, adc_threshold,
                          all_intime_counts, 
                          reco_intime_counts, 
                          reco_outtime_counts,
                          intime_unreco_counts,
                          unreco_fraction, cosmic_reco_fraction );

    output.intime_count_v = reco_intime_counts;
    output.unreco_count_v = intime_unreco_counts;
    output.unreco_fraction_v = unreco_fraction;
    for ( int p=0; p<(int)adc_v.size(); p++ ) {
      output.unreco_fraction_v.push_back( cosmic_reco_fraction[p] );
    }

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
                                                std::vector<int>& all_intime_counts,   // num above thresh pixels intime
                                                std::vector<int>& reco_intime_counts,  // reco pixel intime
                                                std::vector<int>& reco_outtime_counts, // reco pixel outtime
                                                std::vector<int>& intime_unreco_counts, // intime but not recod						
                                                std::vector<float>& unreco_fraction,
                                                std::vector<float>& cosmic_reco_fraction  )
  {

    all_intime_counts.resize(adc_v.size(),0);
    reco_intime_counts.resize(adc_v.size(),0);
    reco_outtime_counts.resize(adc_v.size(),0);
    intime_unreco_counts.resize( adc_v.size(), 0 );
    unreco_fraction.resize( adc_v.size(), 0 );
    cosmic_reco_fraction.resize( adc_v.size(), 0);
    clearVars();
    
    for (int p=0; p<(int)adc_v.size(); p++) {
      all_intime_counts[p]=0;
      reco_intime_counts[p]=0;
      reco_outtime_counts[p]=0;
      intime_unreco_counts[p]=0;
      unreco_fraction[p]=0.;
      cosmic_reco_fraction[p]=0.;

      auto const& img = adc_v[p]; // wire plane image
      auto & mask = numask_v[p];  // has non-zero value where clusters land
      auto const& thrumu = thrumu_v[p]; // has non-zero value for pixels tagged AS COSMIC (untagged are neutrino candidates)
      auto const& meta = adc_v[p].meta();
      if ( meta.rows()!=mask.meta().rows() || meta.cols()!=mask.meta().cols() ) {
        LARCV_CRITICAL() << "dimensions of adc image and nu-candidate mask image are not the same" << std::endl;
      }
      if ( meta.rows()!=thrumu.meta().rows() || meta.cols()!=thrumu.meta().cols() ) {
        LARCV_CRITICAL() << "dimensions of adc image and cosmic-tagged image are not the same" << std::endl;
      }

      
      for (int r=0; r<(int)meta.rows(); r++) {
        for (int c=0; c<(int)meta.cols(); c++) {
          float imgval  = img.pixel(r,c,__FILE__,__LINE__);    // pixel value
          float maskval = mask.pixel(r,c,__FILE__,__LINE__);   // reco pixel
          float tagval  = thrumu.pixel(r,c,__FILE__,__LINE__); // cosmic pixel tag

          if ( imgval>=adc_threshold ) {
            // pixel has above threshold charge
            if ( tagval<10.0 ) {
              // pixel with content and not cosmic-tagged
              all_intime_counts[p]++;

              if ( maskval==0 ) {
                // if mask is zero, we missed it, in principle
                intime_unreco_counts[p]++;
                unreco_fraction[p]++;
                mask.set_pixel(r,c,2.0); /// for debug visualization
              }
	            else {
                // mask value, so a reco sp falls here
	              reco_intime_counts[p]++;
	            }
            }
            else {
              // cosmic/out-of-time tagged pixel
              if ( maskval>0 ) {
                // we recod on top of this apparently
                reco_outtime_counts[p]++;
              }
            }
          }
               
        }
      }

      if ( all_intime_counts[p]>0 ) {
        unreco_fraction[p] /= (float)all_intime_counts[p];
      }
      int total_reco = reco_intime_counts[p]+reco_outtime_counts[p];
      if ( total_reco>0 ) {
        cosmic_reco_fraction[p] = reco_intime_counts[p]/float(total_reco);
      }
      
    }//end of plane loop


    if ( _tree ) {
      // set tree vars
      _intime_count_v = all_intime_counts;
      _unreco_count_v = intime_unreco_counts;
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
    
    std::stringstream ss_intime;
    ss_intime << "  intime counts: ";
    for (auto const& count : all_intime_counts )
      ss_intime << count << " ";
    std::stringstream ss_unreco;
    ss_unreco << "  intime but unreco counts: ";
    for (auto const& count : intime_unreco_counts )
      ss_unreco << count << " ";
    std::stringstream ss_reco;
    ss_reco << "  intime reco counts: ";
    for (auto const& count : reco_intime_counts )
      ss_reco << count << " ";
    std::stringstream ss_frac;
    ss_frac << " frac: ";
    for (auto const& frac : unreco_fraction )
      ss_frac << frac << " ";

    std::stringstream ss_reco_cosmic;
    ss_reco_cosmic << "  out-of-time reco counts: ";
    for (auto const& count : reco_outtime_counts )
      ss_reco_cosmic << count << " ";
    std::stringstream ss_frac_cosmic;
    ss_frac_cosmic << " frac: ";
    for (auto const& frac : cosmic_reco_fraction )
      ss_frac_cosmic << frac << " ";


    LARCV_INFO() << "Results" << std::endl;    
    LARCV_INFO() << ss_intime.str() << std::endl;
    LARCV_INFO() << ss_unreco.str() << std::endl;
    LARCV_INFO() << ss_reco.str() << std::endl;    
    LARCV_INFO() << ss_frac.str() << std::endl;
    LARCV_INFO() << ss_frac_cosmic.str() << std::endl;
    
  }

  void NuSelUnrecoCharge::clearVars()
  {
    _intime_count_v.clear();
    _unreco_count_v.clear();
    _reco_count_v.clear();    
    _unreco_fraction_v.clear();
    _reco_outtime_count_v.clear();
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

  void NuSelUnrecoCharge::analyze_with_spacepoints( larcv::IOManager& iolcv,
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
      = { "trackprojsplit_wcfilter", 
          "showerkp",
          "showergoodhit" };

    larcv::EventImage2D* ev_img
      = (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D, "wire" );
    auto const& adc_v = ev_img->as_vector();

    std::map< int, int > idxhit_v;
    int nhits = 0;
    for (auto& producer : cluster_producers ) {
      larlite::event_larflowcluster* ev_cluster
        = (larlite::event_larflowcluster*)ioll.get_data(larlite::data::kLArFlowCluster, producer);
      for ( auto const& cluster : *(ev_cluster) ) {
        for ( auto const& hit : cluster ) {
          idxhit_v[ hit.idxhit ] = 0;
          nhits++;
        }
      }
    }

    larlite::event_larflow3dhit* ev_kpvetoed 
      = (larlite::event_larflow3dhit*)ioll.get_data(larlite::data::kLArFlow3DHit, "projsplitvetoed");
    for ( auto const& hit : *(ev_kpvetoed) ) {
      idxhit_v[ hit.idxhit ] = 0;
      nhits++;
    }
    LARCV_INFO() << "nhits=" << nhits << "  idxhit_v.size()=" << idxhit_v.size() << std::endl;

    // now we ask, did we use the hit?
    int nfound_track = 0;
    for ( auto& trackcluster : nuvtx.track_hitcluster_v ) {
      for ( auto& trackhit : trackcluster ) {
        auto it = idxhit_v.find( trackhit.idxhit );
        if ( it==idxhit_v.end() ) {
          LARCV_INFO() << "  trackhit not in original hit map. idxhit=" << trackhit.idxhit << std::endl;
        }
        else {
          // set value to 1, to indicate it was found.
          it->second = 1;
          nfound_track++;
        }
      }
    }
    int nfound_shower = 0;
    for ( auto& shower : nuvtx.shower_v ) {
      for ( auto& hit : shower ) {
        auto it = idxhit_v.find( hit.idxhit );
        if ( it==idxhit_v.end() ) {
          LARCV_INFO() << "  shower hit not in original hit map. idxhit=" << hit.idxhit << std::endl;
        }
        else {
          // set value to 1, to indicate it was found.
          it->second = 1;
          nfound_shower++;
        }
      }
    }

    output.intime_count_v.push_back( nhits );
    output.unreco_count_v.push_back( nhits-(nfound_track+nfound_shower));
    if ( nhits>0 )
      output.unreco_fraction_v.push_back( (nhits-(nfound_track+nfound_shower))/float(nhits) );
    else
      output.unreco_fraction_v.push_back( 0.0 );

    LARCV_INFO() << "Results [spacepoint results appended at end]" << std::endl;
    std::stringstream ss_intime;
    ss_intime << "  intime counts: ";
    for (auto const& count : output.intime_count_v )
      ss_intime << count << " ";
    std::stringstream ss_unreco;
    ss_unreco << "  unreco counts: ";
    for (auto const& count : output.unreco_count_v )
      ss_unreco << count << " ";
    std::stringstream ss_reco;
    ss_reco << "  reco counts: ntrack=" << nfound_track << " nshower=" << nfound_shower;
    std::stringstream ss_frac;
    for (auto const& frac : output.unreco_fraction_v )
      ss_frac << frac << " ";
    
    LARCV_INFO() << ss_intime.str() << std::endl;
    LARCV_INFO() << ss_unreco.str() << std::endl;
    LARCV_INFO() << ss_reco.str() << std::endl;    
    LARCV_INFO() << ss_frac.str() << std::endl;
    
  }
  
  
}
}
