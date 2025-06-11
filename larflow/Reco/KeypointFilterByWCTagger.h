#ifndef __KEYPOINT_FILTER_BY_WC_TAGGER_H__
#define __KEYPOINT_FILTER_BY_WC_TAGGER_H__

#include <vector>
#include <string>

#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/Image2D.h"
#include "larcv/core/DataFormat/EventImage2D.h"
#include "larlite/DataFormat/storage_manager.h"
#include "larlite/DataFormat/larflow3dhit.h"

namespace larflow {
namespace reco {

  /**
   * @ingroup Reco
   * @class KeypointFilterByWCTagger
   * @brief Filters LArMatch hits and keypoints using WireCell cosmic ray tagger
   *
   * This class separates in-time (neutrino-induced) hits from cosmic ray activity
   * using the WireCell cosmic ray tagger. The tagger produces "thrumu" images where
   * pixels with high values (>5) indicate cosmic ray activity.
   *
   * The algorithm aims to preserve neutrino interactions while removing cosmic rays:
   * 1. For each 3D hit/keypoint, project to 2D wire plane coordinates
   * 2. Check WireCell tagger value at projected location(s)
   * 3. If cosmic-tagged (value > 5), check SSNet shower score
   * 4. Keep cosmic-tagged pixels if they have high shower score (EM activity)
   * 5. Reject if cosmic-tagged with low shower score (muon-like activity)
   *
   * Key features:
   * - Preserves electron shower pixels to maintain νe efficiency
   * - Uses multi-plane consensus for robust filtering
   * - Separate algorithms for hits vs keypoints (different spatial windows)
   * - Configurable input/output tree names
   *
   * Physics motivation:
   * - Cosmic muons appear as long tracks across multiple planes
   * - Neutrino interactions are localized and often contain EM showers
   * - Cosmic electrons/photons should be kept for νe analysis
   *
   * Usage patterns:
   * - process(): Filter both hits and keypoints
   * - process_hits(): Filter only LArMatch spacepoints
   * - process_keypoints(): Filter only vertex candidates
   */
  class KeypointFilterByWCTagger : public larcv::larcv_base {

  public:
    
    /**
     * @brief Default constructor with standard parameter values
     *
     * Initializes with commonly used default values for MicroBooNE:
     * - Input trees: "larmatch", "keypoint", "wire", "thrumu"
     * - SSNet images: "ubspurn_plane" stem (expects plane0/1/2 suffix)
     * - Output trees: "taggerfilterhit", "taggerrejecthit", "taggerfilterkeypoint"
     * - Rejected hits not saved by default
     */
    KeypointFilterByWCTagger();

    /**
     * @brief Virtual destructor
     */
    virtual ~KeypointFilterByWCTagger() {};


    /**
     * @brief Process both hits and keypoints using I/O managers
     *
     * High-level interface that filters both LArMatch spacepoints and keypoints
     * using WireCell cosmic tagger. This is the most commonly used entry point.
     *
     * @param iolcv LArCV I/O manager (reads wire images, tagger images, SSNet scores)
     * @param ioll larlite I/O manager (reads hits/keypoints, writes filtered results)
     *
     * This method calls process_hits() followed by process_keypoints().
     */
    void process( larcv::IOManager& iolcv, larlite::storage_manager& ioll );

    /**
     * @brief Filter LArMatch spacepoints using WireCell tagger
     *
     * Separates LArMatch 3D spacepoints into in-time (neutrino candidate) and
     * out-of-time (cosmic) categories using WireCell cosmic ray tagger and
     * SSNet shower scores.
     *
     * Algorithm:
     * 1. Project each 3D hit to wire plane coordinates
     * 2. Check WireCell tagger value at each projection
     * 3. If cosmic-tagged (>5), check SSNet shower score
     * 4. Reject if cosmic-tagged with low shower score on ≥2 planes
     *
     * @param iolcv LArCV I/O manager (reads wire, tagger, SSNet images)
     * @param ioll larlite I/O manager (reads hits, writes filtered results)
     *
     * Output trees:
     * - Filtered hits saved to _output_filteredhits_tree_name
     * - Rejected hits saved to _output_rejectedhits_tree_name (if enabled)
     */
    void process_hits( larcv::IOManager& iolcv, larlite::storage_manager& ioll );

    /**
     * @brief Filter keypoints using WireCell tagger
     *
     * Filters keypoint candidates by checking for cosmic ray activity in
     * an 11x11 pixel window around each projected keypoint location.
     *
     * Algorithm:
     * 1. Project 3D keypoint to wire plane coordinates
     * 2. Search 11x11 window around projection on each plane
     * 3. Count cosmic-tagged pixels and shower-like pixels
     * 4. Reject if cosmic activity without sufficient shower activity on ≥2 planes
     *
     * @param iolcv LArCV I/O manager (reads wire, tagger, SSNet images)
     * @param ioll larlite I/O manager (reads keypoints, writes filtered results)
     *
     * Output tree: Filtered keypoints saved to _output_keypoint_tree_name
     */
    void process_keypoints( larcv::IOManager& iolcv, larlite::storage_manager& ioll );    

    /**
     * @brief Standalone function to filter LArMatch hits using cosmic tagger
     *
     * Core algorithm for filtering 3D spacepoints based on WireCell cosmic
     * tagging and SSNet shower scores. This can be used independently of
     * the I/O framework.
     *
     * Decision criteria per hit:
     * - Project to wire planes, check tagger value at each projection
     * - If tagger > 5 and SSNet shower score < 0.5: cosmic muon candidate
     * - Reject if cosmic muon candidate on ≥2 planes
     *
     * @param adc_v Vector of wire plane ADC images (for geometry/projection)
     * @param tagged_v Vector of WireCell cosmic tagger images
     * @param shower_ssnet_v Vector of SSNet shower score images
     * @param hit_v Input collection of LArMatch 3D hits to filter
     * @param kept_v Output decision vector (1=keep, 0=reject, same size as hit_v)
     */
    void filter_larmatchhits_using_tagged_image( const std::vector<larcv::Image2D>& adc_v,
                                                 const std::vector<larcv::Image2D>& tagged_v,
                                                 const std::vector< const larcv::Image2D* >& shower_ssnet_v,
                                                 const std::vector<larlite::larflow3dhit>& hit_v,
                                                 std::vector<int>& kept_v );

    /**
     * @brief Standalone function to filter keypoints using cosmic tagger
     *
     * Core algorithm for filtering keypoint candidates using WireCell cosmic
     * tagging with spatial window analysis. More sophisticated than hit filtering
     * due to spatial extent consideration.
     *
     * Decision criteria per keypoint:
     * - Project to wire planes, analyze 11x11 window around each projection
     * - Count cosmic-tagged pixels (tagger > 5) in window
     * - Count likely shower pixels (shower score > 0.25) among cosmic-tagged
     * - Reject if cosmic activity without showers on ≥2 planes
     *
     * @param adc_v Vector of wire plane ADC images (for geometry/projection)
     * @param tagged_v Vector of WireCell cosmic tagger images
     * @param shower_ssnet_v Vector of SSNet shower score images
     * @param keypoint_v Input collection of keypoint candidates to filter
     * @param kept_v Output decision vector (1=keep, 0=reject, same size as keypoint_v)
     */
    void filter_keypoint_using_tagged_image( const std::vector<larcv::Image2D>& adc_v,
                                             const std::vector<larcv::Image2D>& tagged_v,
                                             const std::vector< const larcv::Image2D* >& shower_ssnet_v,
                                             const std::vector<larlite::larflow3dhit>& keypoint_v,
                                             std::vector<int>& kept_v );
    

  protected:

    // Configuration parameters
    std::string _input_keypoint_tree_name;       ///< Name of input keypoint tree (default: "keypoint")
    std::string _input_larmatch_tree_name;       ///< Name of input LArMatch spacepoint tree (default: "larmatch")
    std::string _input_adc_tree_name;            ///< Name of wire plane ADC image tree (default: "wire")
    std::string _input_taggerimg_tree_name;      ///< Name of WireCell cosmic tagger image tree (default: "thrumu")
    std::string _ssnet_stem_name;                ///< Stem name for SSNet score images (default: "ubspurn_plane")
    std::string _output_keypoint_tree_name;      ///< Name of output filtered keypoint tree (default: "taggerfilterkeypoint")
    std::string _output_filteredhits_tree_name;  ///< Name of output in-time hit tree (default: "taggerfilterhit")
    std::string _output_rejectedhits_tree_name;  ///< Name of output cosmic hit tree (default: "taggerrejecthit")
    bool        _save_rejected_hits;             ///< Whether to save cosmic-tagged hits (default: false)

    /**
     * @brief Initialize configuration parameters to default values
     *
     * Sets standard MicroBooNE naming conventions:
     * - Input trees: "keypoint", "larmatch", "wire", "thrumu"
     * - SSNet stem: "ubspurn_plane" (expects plane0/1/2 suffix)
     * - Output trees: "taggerfilterkeypoint", "taggerfilterhit", "taggerrejecthit"
     * - Rejected hits not saved by default (can be large)
     */
    void set_defaults() {
      _input_keypoint_tree_name = "keypoint";
      _input_larmatch_tree_name = "larmatch";      
      _input_adc_tree_name = "wire";
      _input_taggerimg_tree_name = "thrumu";
      _ssnet_stem_name = "ubspurn_plane";
      _output_keypoint_tree_name = "taggerfilterkeypoint";
      _output_filteredhits_tree_name = "taggerfilterhit";
      _output_rejectedhits_tree_name = "taggerrejecthit";
      _save_rejected_hits = false;
    };


  public:

    // Configuration setters

    /**
     * @brief Set name of input keypoint tree
     * @param keypoint Tree name containing keypoint candidates (larflow3dhit format)
     *
     * Keypoints are typically vertex, track end, or shower start candidates
     * from the keypoint detection network.
     */
    void set_input_keypoint_tree_name( std::string keypoint )  { _input_keypoint_tree_name=keypoint; };

    /**
     * @brief Set name of input LArMatch spacepoint tree
     * @param larmatch Tree name containing LArMatch 3D spacepoints (larflow3dhit format)
     *
     * These are the raw 3D spacepoints from the LArMatch network before
     * any cosmic ray filtering.
     */
    void set_input_larmatch_tree_name( std::string larmatch )  { _input_larmatch_tree_name=larmatch; };

    /**
     * @brief Set name of WireCell cosmic tagger image tree
     * @param tagger Tree name containing WireCell "thrumu" cosmic tag images
     *
     * WireCell produces images where pixel values > 5 indicate cosmic ray activity.
     * This is the primary discriminant for separating cosmic vs neutrino activity.
     */
    void set_input_taggerimg_tree_name( std::string tagger )   { _input_taggerimg_tree_name=tagger; };

    /**
     * @brief Set stem name for SSNet score image trees
     * @param stem Base name for SSNet images (e.g., "ubspurn_plane" → "ubspurn_plane0", "ubspurn_plane1", "ubspurn_plane2")
     *
     * SSNet provides track vs shower classification to help distinguish
     * cosmic muons (track-like) from cosmic electrons/photons (shower-like).
     */
    void set_input_ssnet_stem_name( std::string stem )         { _ssnet_stem_name=stem; };

    /**
     * @brief Set output tree name for filtered keypoints
     * @param keypoint Tree name for keypoints that pass cosmic filter
     *
     * Filtered keypoints are candidates for neutrino vertex reconstruction.
     */
    void set_output_keypoint_tree_name( std::string keypoint ) { _output_keypoint_tree_name=keypoint; };

    /**
     * @brief Set output tree name for in-time (neutrino candidate) hits
     * @param hits Tree name for hits that pass cosmic filter
     *
     * These hits are used for downstream neutrino reconstruction.
     */
    void set_output_filteredhits_tree_name( std::string hits ) { _output_filteredhits_tree_name=hits; };

    /**
     * @brief Set output tree name for cosmic-tagged (rejected) hits
     * @param hits Tree name for hits identified as cosmic ray activity
     *
     * Only saved if set_save_rejected_hits(true) is called. Can be useful
     * for cosmic ray studies or algorithm debugging.
     */
    void set_output_rejectedhits_tree_name( std::string hits ) { _output_rejectedhits_tree_name=hits; };

    /**
     * @brief Enable/disable saving of cosmic-tagged hits
     * @param save If true, save rejected hits to output tree (default: false)
     *
     * Cosmic hits can be numerous, so saving them is optional and disabled
     * by default to reduce output file size.
     */
    void set_save_rejected_hits( bool save )                   { _save_rejected_hits=save; };
    
  };

}
}

#endif
