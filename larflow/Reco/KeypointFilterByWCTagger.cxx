#include "KeypointFilterByWCTagger.h"

#include "larlite/LArUtil/LArProperties.h"
#include "larlite/LArUtil/Geometry.h"
#include "larlite/DataFormat/pcaxis.h"

namespace larflow {
namespace reco {

  /**
   * @brief Default constructor with standard parameter initialization
   *
   * Initializes the cosmic ray filter with commonly used MicroBooNE defaults:
   * - Input trees: "larmatch", "keypoint", "wire", "thrumu"
   * - SSNet images: "ubspurn_plane" stem (expects plane0/1/2 suffix)
   * - Output trees: "taggerfilterhit", "taggerrejecthit", "taggerfilterkeypoint"
   * - Rejected hits not saved by default to reduce output size
   */
  KeypointFilterByWCTagger::KeypointFilterByWCTagger()
    : larcv::larcv_base("KeypointFilterByWCTagger")
  {
    set_defaults();
  }


  /**
   * @brief High-level interface to filter both hits and keypoints using cosmic tagger
   *
   * This is the main entry point for WireCell cosmic ray filtering. It processes
   * both LArMatch spacepoints and keypoint candidates in sequence, applying
   * the cosmic ray tagging algorithm to separate in-time (neutrino candidate)
   * activity from cosmic ray background.
   *
   * Algorithm flow:
   * 1. First filters LArMatch hits using process_hits()
   * 2. Then filters keypoint candidates using process_keypoints()
   * 3. Results saved to configured output trees
   *
   * @param[in] iolcv LArCV I/O manager containing wire images, tagger images, and SSNet scores
   * @param[in] ioll  larlite I/O manager containing input hits/keypoints and output containers
   */
  void KeypointFilterByWCTagger::process( larcv::IOManager& iolcv,
                                          larlite::storage_manager& ioll )
  {
    // Process both data types using the same algorithm framework
    process_hits( iolcv, ioll );
    process_keypoints( iolcv, ioll );
  }

  /**
   * @brief Filter LArMatch spacepoints using WireCell cosmic ray tagger and SSNet scores
   *
   * This method separates LArMatch 3D spacepoints into in-time (neutrino candidate)
   * and cosmic ray categories using the WireCell cosmic ray tagger combined with
   * SSNet shower scores to preserve electromagnetic activity.
   *
   * Algorithm:
   * 1. Load wire plane images for 3D-to-2D projection
   * 2. Load WireCell cosmic tagger images ("thrumu")
   * 3. Load SSNet shower score images for each plane
   * 4. Apply cosmic filtering logic via filter_larmatchhits_using_tagged_image()
   * 5. Separate hits into in-time vs cosmic containers
   *
   * Physics motivation: Cosmic muons appear as long tracks tagged by WireCell,
   * but cosmic electrons/photons should be preserved for νe efficiency.
   *
   * @param[in] iolcv LArCV I/O manager containing images needed for filtering
   * @param[in] ioll  larlite I/O manager containing hits and output containers
   */
  void KeypointFilterByWCTagger::process_hits( larcv::IOManager& iolcv,
                                               larlite::storage_manager& ioll )
  {
    // Load wire plane ADC images for 3D to 2D projection geometry
    larcv::EventImage2D* ev_adc_v =
      (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, _input_adc_tree_name );
    const std::vector<larcv::Image2D>& adc_v = ev_adc_v->Image2DArray();    
    LARCV_INFO() << "Input wire images [" << _input_adc_tree_name << "]: " << adc_v.size() << std::endl;
    
    // Load WireCell cosmic ray tagger images (pixel values >5 indicate cosmic activity)
    larcv::EventImage2D* ev_tagger =
      (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, _input_taggerimg_tree_name );
    const std::vector<larcv::Image2D>& tagged_v = ev_tagger->Image2DArray();
    LARCV_INFO() << "Input tagged images [" << _input_taggerimg_tree_name << "]: " << tagged_v.size() << std::endl;

    // Load LArMatch 3D spacepoints to be filtered
    LARCV_INFO() << "Input larmatch hit tree: " << _input_larmatch_tree_name << std::endl;
    larlite::event_larflow3dhit* ev_larmatch =
      (larlite::event_larflow3dhit*)ioll.get_data( larlite::data::kLArFlow3DHit, _input_larmatch_tree_name );
    
    // Load SSNet shower score images for each wire plane
    // These help distinguish cosmic muons (track-like) from cosmic e/γ (shower-like)
    larcv::EventImage2D* ev_ssnet_v[3] = {nullptr};
    for ( size_t p=0; p<3; p++ ) {
      char prodname[20];
      sprintf( prodname, "%s%d", _ssnet_stem_name.c_str(), (int)p );
      ev_ssnet_v[p] = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, prodname );
    }

    // Collect pointers to shower score images for algorithm
    std::vector<const larcv::Image2D*> ssnet_showerimg_v;
    for ( size_t p=0; p<3; p++ )
      ssnet_showerimg_v.push_back(&(ev_ssnet_v[p]->Image2DArray()[0]));

    // Apply cosmic ray filtering algorithm
    std::vector<int> kept_hit_v( ev_larmatch->size(), 0 );
    filter_larmatchhits_using_tagged_image( adc_v, tagged_v, ssnet_showerimg_v, *ev_larmatch, kept_hit_v );
    
    // Prepare output containers
    larlite::event_larflow3dhit* ev_filteredhits_output = 
      (larlite::event_larflow3dhit*)ioll.get_data( larlite::data::kLArFlow3DHit, _output_filteredhits_tree_name );

    larlite::event_larflow3dhit* ev_rejectedhits_output = nullptr;
    if ( _save_rejected_hits ) {
      ev_rejectedhits_output =
        (larlite::event_larflow3dhit*)ioll.get_data( larlite::data::kLArFlow3DHit, _output_rejectedhits_tree_name );
    }

    // Separate hits based on filtering decision
    for ( size_t ihit=0; ihit<ev_larmatch->size(); ihit++ ) {
      if ( kept_hit_v[ihit]==1 ) {
        ev_filteredhits_output->push_back( ev_larmatch->at(ihit) );  // In-time (neutrino candidate)
      }
      else if ( _save_rejected_hits && kept_hit_v[ihit]==0 ) {
        ev_rejectedhits_output->push_back( ev_larmatch->at(ihit) );  // Cosmic ray tagged
      }
    }

    // Performance logging
    LARCV_INFO() << "Filtered hits: " << ev_filteredhits_output->size() << " in-time / " 
                 << ev_larmatch->size() << " total (" 
                 << 100.0*ev_filteredhits_output->size()/ev_larmatch->size() << "% kept)" << std::endl;
    if ( _save_rejected_hits ) {
      LARCV_INFO() << "Rejected hits: " << ev_rejectedhits_output->size() << " cosmic-tagged" << std::endl;
    }
  }

  /**
   * @brief Filter keypoint candidates using WireCell cosmic ray tagger with spatial window analysis
   *
   * This method filters vertex, track end, and shower start candidates using a more
   * sophisticated approach than hit filtering. It analyzes an 11x11 pixel window
   * around each keypoint's projected location to assess cosmic ray contamination
   * while preserving electromagnetic activity that could indicate real vertices.
   *
   * Algorithm:
   * 1. Load wire plane images and cosmic tagger data
   * 2. Load keypoint candidates and associated PCA axis information
   * 3. Load SSNet shower scores for cosmic muon vs e/γ discrimination
   * 4. Apply spatial window filtering via filter_keypoint_using_tagged_image()
   * 5. Save filtered keypoints with corresponding PCA data
   *
   * The spatial window approach accounts for the fact that true neutrino vertices
   * may have some cosmic contamination nearby but should have significant
   * electromagnetic shower activity.
   *
   * @param[in] iolcv LArCV I/O manager containing images needed for filtering
   * @param[in] ioll  larlite I/O manager containing keypoints and output containers
   */
  void KeypointFilterByWCTagger::process_keypoints( larcv::IOManager& iolcv,
                                                    larlite::storage_manager& ioll )
  {
    // Load wire plane ADC images for 3D to 2D projection geometry
    larcv::EventImage2D* ev_adc_v =
      (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, _input_adc_tree_name );
    const std::vector<larcv::Image2D>& adc_v = ev_adc_v->Image2DArray();    
    LARCV_INFO() << "Input wire images [" << _input_adc_tree_name << "]: " << adc_v.size() << std::endl;
    
    // Load WireCell cosmic ray tagger images
    larcv::EventImage2D* ev_tagger =
      (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, _input_taggerimg_tree_name );
    const std::vector<larcv::Image2D>& tagged_v = ev_tagger->Image2DArray();
    LARCV_INFO() << "Input tagged images [" << _input_taggerimg_tree_name << "]: " << tagged_v.size() << std::endl;

    // Load keypoint candidates (vertices, track ends, shower starts) and associated PCA data
    LARCV_INFO() << "Input keypoint tree: " << _input_keypoint_tree_name << std::endl;    
    larlite::event_larflow3dhit* ev_keypoint =
      (larlite::event_larflow3dhit*)ioll.get_data( larlite::data::kLArFlow3DHit, _input_keypoint_tree_name );
    larlite::event_pcaxis* ev_keypoint_pcaxis =
      (larlite::event_pcaxis*)ioll.get_data( larlite::data::kPCAxis, _input_keypoint_tree_name );
    
    // Load SSNet shower score images for each wire plane
    // Critical for distinguishing cosmic muons from electromagnetic activity
    larcv::EventImage2D* ev_ssnet_v[3] = {nullptr};
    for ( size_t p=0; p<3; p++ ) {
      char prodname[20];
      sprintf( prodname, "%s%d", _ssnet_stem_name.c_str(), (int)p );
      ev_ssnet_v[p] = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, prodname );
    }

    // Collect pointers to shower score images
    std::vector<const larcv::Image2D*> ssnet_showerimg_v;
    for ( size_t p=0; p<3; p++ )
      ssnet_showerimg_v.push_back(&(ev_ssnet_v[p]->Image2DArray()[0]));

    // Apply spatial window cosmic ray filtering algorithm
    std::vector<int> kept_keypoint_v( ev_keypoint->size(), 0 );
    filter_keypoint_using_tagged_image( adc_v, tagged_v, ssnet_showerimg_v, *ev_keypoint, kept_keypoint_v );
    
    // Prepare output containers for filtered keypoints and their PCA data
    larlite::event_larflow3dhit* ev_keypoint_output = 
      (larlite::event_larflow3dhit*)ioll.get_data( larlite::data::kLArFlow3DHit, _output_keypoint_tree_name );
    larlite::event_pcaxis* ev_kpaxis_output = 
      (larlite::event_pcaxis*)ioll.get_data( larlite::data::kPCAxis, _output_keypoint_tree_name );

    // Save keypoints that pass cosmic ray filter along with their PCA axis data
    for ( size_t ikp=0; ikp<ev_keypoint->size(); ikp++ ) {
      if ( kept_keypoint_v[ikp]==1 )  {
        ev_keypoint_output->push_back( ev_keypoint->at(ikp) );
        ev_kpaxis_output->push_back( ev_keypoint_pcaxis->at(ikp) );
      }
    }

    // Performance logging
    LARCV_INFO() << "Filtered keypoints: " << ev_keypoint_output->size() << " in-time / " 
                 << ev_keypoint->size() << " total (" 
                 << 100.0*ev_keypoint_output->size()/ev_keypoint->size() << "% kept)" << std::endl;
  }
  

  /**
   * @brief Core algorithm for filtering LArMatch spacepoints using WireCell cosmic tagging
   *
   * This is the standalone implementation of cosmic ray filtering for 3D spacepoints.
   * The algorithm is designed to remove cosmic muons while preserving electromagnetic
   * shower activity to maintain efficiency for electron neutrino analysis.
   *
   * Decision logic per hit:
   * 1. Project 3D hit to 2D wire plane coordinates using stored targetwire values
   * 2. For each plane, check WireCell tagger value at projected pixel
   * 3. If tagged as cosmic (value > 5), check SSNet shower score
   * 4. Count planes where hit is cosmic-tagged with low shower score (< 0.5)
   * 5. Reject hit if cosmic muon signature found on ≥2 planes
   *
   * Physics motivation:
   * - Cosmic muons appear as long tracks across multiple wire planes
   * - Cosmic electrons/photons have high shower scores and should be kept for νe
   * - Multi-plane consensus reduces false positives from single-plane noise
   *
   * @param[in] adc_v Wire plane ADC images (for geometry/bounds checking)
   * @param[in] tagged_v WireCell cosmic tagger images (values >5 indicate cosmic activity)
   * @param[in] shower_ssnet_v SSNet shower score images (values >0.5 indicate EM showers)
   * @param[in] larmatch_v Input LArMatch 3D spacepoints to filter
   * @param[out] kept_v Decision vector (1=keep, 0=reject, same size as larmatch_v)
   */
  void KeypointFilterByWCTagger::filter_larmatchhits_using_tagged_image( const std::vector<larcv::Image2D>& adc_v,
                                                                         const std::vector<larcv::Image2D>& tagged_v,
                                                                         const std::vector< const larcv::Image2D* >& shower_ssnet_v,
                                                                         const std::vector<larlite::larflow3dhit>& larmatch_v,
                                                                         std::vector<int>& kept_v )
  {
    // Initialize output decision vector
    kept_v.clear();
    kept_v.resize( larmatch_v.size(), 0 );
    
    // Process each 3D spacepoint
    for ( size_t ihit=0; ihit<larmatch_v.size(); ihit++ ) {
      auto const& hit = larmatch_v[ihit];
      
      // Bounds check: ensure hit is within time window of wire plane images
      if ( hit.tick <= adc_v[0].meta().min_y() || hit.tick >= adc_v[0].meta().max_y() ) {
        kept_v[ihit] = 0;  // Reject hits outside time bounds
        continue;
      }
      
      // Convert time coordinate to row index (consistent across all planes)
      int row = adc_v[0].meta().row( hit.tick+1, __FILE__, __LINE__ );
      
      // Count planes where hit appears as cosmic muon (tagged cosmic + low shower score)
      int nplanes_tagged = 0;
      
      // Check cosmic tagging status on each wire plane
      for ( size_t p=0; p<adc_v.size(); p++ ) {
        // Convert wire coordinate to column index for this plane
        int col = adc_v[p].meta().col( hit.targetwire[p], __FILE__, __LINE__ );
        
        // Check WireCell cosmic tagger value at projected pixel location
        int tagged = tagged_v[p].pixel( row, col );
        
        if ( tagged > 5 ) {  // Pixel flagged as cosmic by WireCell
          
          // Check SSNet shower score to distinguish muons from EM activity
          float shower_score = 0.0;
          if ( shower_ssnet_v.size() > 0 ) {
            shower_score = shower_ssnet_v[p]->pixel(row, col);
          }
          
          // Low shower score + cosmic tag = likely cosmic muon
          if ( shower_score < 0.5 ) {
            nplanes_tagged++;
          }
          // High shower score means EM activity - keep for νe efficiency
        }        
      }
      
      // Apply multi-plane consensus: reject if cosmic muon signature on ≥2 planes
      if ( nplanes_tagged >= 2 ) {
        kept_v[ihit] = 0;  // Reject: cosmic muon candidate
      }
      else {
        kept_v[ihit] = 1;  // Keep: in-time or EM activity
      }
    }
  }


  /**
   * @brief Core algorithm for filtering keypoint candidates using spatial window analysis
   *
   * This sophisticated algorithm filters vertex, track end, and shower start candidates
   * by analyzing an 11x11 pixel neighborhood around each keypoint's projected location.
   * The spatial approach accounts for the extended nature of neutrino vertices while
   * rejecting keypoints that lie on cosmic muon tracks.
   *
   * Decision logic per keypoint:
   * 1. Convert 3D position to 2D projections using LArUtil geometry
   * 2. For each wire plane, analyze 11x11 window around projected location
   * 3. Count cosmic-tagged pixels (tagger > 5) within window
   * 4. Count likely shower pixels (SSNet score < 0.25) among cosmic-tagged pixels
   * 5. Tag plane if cosmic activity present but minimal shower activity (≤2 shower pixels)
   * 6. Reject keypoint if tagged on ≥2 planes (multi-plane cosmic consensus)
   *
   * Physics motivation:
   * - True neutrino vertices may have nearby cosmic contamination but should contain
   *   significant electromagnetic shower activity from neutrino interaction products
   * - Cosmic muon tracks appear as extended features with minimal shower activity
   * - The 11x11 window captures vertex extent while maintaining spatial precision
   *
   * Algorithm parameters:
   * - Window size: 11x11 pixels (±5 around center)
   * - Cosmic threshold: WireCell tagger value > 5
   * - Shower threshold: SSNet score < 0.25 (more restrictive than hit filtering)
   * - Shower pixel limit: ≤2 shower pixels in cosmic-tagged region
   *
   * @param[in] adc_v Wire plane ADC images (for geometry and bounds checking)
   * @param[in] tagged_v WireCell cosmic tagger images (values >5 indicate cosmic activity)
   * @param[in] shower_ssnet_v SSNet shower score images (lower scores = more track-like)
   * @param[in] keypoint_v Input keypoint candidates to filter
   * @param[out] kept_v Decision vector (1=keep, 0=reject, same size as keypoint_v)
   */
  void KeypointFilterByWCTagger::filter_keypoint_using_tagged_image( const std::vector<larcv::Image2D>& adc_v,
                                                                     const std::vector<larcv::Image2D>& tagged_v,
                                                                     const std::vector< const larcv::Image2D* >& shower_ssnet_v,
                                                                     const std::vector<larlite::larflow3dhit>& keypoint_v,
                                                                     std::vector<int>& kept_v )
  {
    // Initialize output decision vector
    kept_v.clear();
    kept_v.resize( keypoint_v.size(), 0 );
    
    // Process each keypoint candidate
    for ( size_t ihit=0; ihit<keypoint_v.size(); ihit++ ) {
      auto const& hit = keypoint_v[ihit];

      // Convert 3D X coordinate to drift time (tick) using detector properties
      // X=0 corresponds to tick=3200 in MicroBooNE coordinates
      int tick = hit[0]/larutil::LArProperties::GetME()->DriftVelocity()/0.5 + 3200;
      
      // Bounds check: ensure keypoint is within time window of wire plane images
      if ( tick <= adc_v[0].meta().min_y() || tick >= adc_v[0].meta().max_y() ) {
        kept_v[ihit] = 0;  // Reject keypoints outside time bounds
        continue;
      }
      
      // Convert time coordinate to row index (consistent across all planes)
      int row = adc_v[0].meta().row( tick, __FILE__, __LINE__ );
      
      // 3D position for wire plane projection
      std::vector<double> dpos = { hit[0], hit[1], hit[2] };

      // Count planes where keypoint neighborhood shows cosmic muon signature
      int nplanes_tagged = 0;
      
      // Analyze spatial neighborhood on each wire plane
      for ( size_t p=0; p<adc_v.size(); p++ ) {
        
        // Project 3D position to wire coordinate for this plane
        int wire = larutil::Geometry::GetME()->NearestWire( dpos, p );
        int col = adc_v[p].meta().col( wire );

        // Counters for spatial window analysis
        int nshower = 0;   // Number of likely shower pixels among cosmic-tagged
        int ntagged = 0;   // Total number of cosmic-tagged pixels in window

        // Search 11x11 window around projected keypoint location
        for (int dr=-5; dr<=5; dr++) {
          int r = row + dr;
          if ( r < 0 || r >= (int)adc_v[p].meta().rows() ) continue;  // Bounds check
          
          for (int dc=-5; dc<=5; dc++) {
            int c = col + dc;
            if ( c < 0 || c >= (int)adc_v[p].meta().cols() ) continue;  // Bounds check

            // Check WireCell cosmic tagger value at this pixel
            int tagged = tagged_v[p].pixel( r, c );
            
            if ( tagged > 5 ) {  // Pixel flagged as cosmic by WireCell
              ntagged++;
              
              // Check SSNet shower score to identify electromagnetic activity
              float shower_score = shower_ssnet_v[p]->pixel(r, c);

              // Count pixels with low shower score (track-like among cosmic-tagged)
              if ( shower_score < 0.25 ) {
                nshower++;
              }
            }
          } // column neighborhood loop
        } // row neighborhood loop

        // Tag this plane if cosmic activity present with minimal shower activity
        // This indicates cosmic muon track rather than neutrino vertex with EM showers
        if ( ntagged > 0 && nshower <= 2 ) {
          nplanes_tagged++;
        }
        
      } // wire plane loop
        
      // Apply multi-plane consensus: reject if cosmic muon signature on ≥2 planes
      if ( nplanes_tagged >= 2 ) {
        kept_v[ihit] = 0;  // Reject: likely on cosmic muon track
      }
      else {
        kept_v[ihit] = 1;  // Keep: likely neutrino vertex candidate
      }
      
    } // keypoint loop
  }
  
  
}
}


