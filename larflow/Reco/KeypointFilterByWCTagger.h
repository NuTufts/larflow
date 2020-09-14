#ifndef __KEYPOINT_FILTER_BY_WC_TAGGER_H__
#define __KEYPOINT_FILTER_BY_WC_TAGGER_H__

#include <vector>
#include <string>

#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/Image2D.h"
#include "larcv/core/DataFormat/EventImage2D.h"
#include "DataFormat/storage_manager.h"
#include "DataFormat/larflow3dhit.h"

namespace larflow {
namespace reco {

  /**
   * @ingroup Reco
   * @class KeypointFilterByWCTagger
   * @brief Filter out keypoints based on proximity to wire cell-tagged pixels
   *
   * For each keypoint, project the 3D position into the image.
   * Look for wire cell pixels tagged as cosmic.
   * For those near wire cell tagged pixels, remove them as cosmic.
   *
   */
  class KeypointFilterByWCTagger : public larcv::larcv_base {

  public:
    
    KeypointFilterByWCTagger();
    virtual ~KeypointFilterByWCTagger() {};


    void process( larcv::IOManager& iolcv, larlite::storage_manager& ioll );
    void process_hits( larcv::IOManager& iolcv, larlite::storage_manager& ioll );
    void process_keypoints( larcv::IOManager& iolcv, larlite::storage_manager& ioll );    

    void filter_larmatchhits_using_tagged_image( const std::vector<larcv::Image2D>& adc_v,
                                                 const std::vector<larcv::Image2D>& tagged_v,
                                                 const std::vector< const larcv::Image2D* >& shower_ssnet_v,
                                                 const std::vector<larlite::larflow3dhit>& hit_v,
                                                 std::vector<int>& kept_v );

    void filter_keypoint_using_tagged_image( const std::vector<larcv::Image2D>& adc_v,
                                             const std::vector<larcv::Image2D>& tagged_v,
                                             const std::vector< const larcv::Image2D* >& shower_ssnet_v,
                                             const std::vector<larlite::larflow3dhit>& keypoint_v,
                                             std::vector<int>& kept_v );
    

  protected:

    //parameters
    std::string _input_keypoint_tree_name;       ///< name of tree with keypoints to filter [default:keypoint]
    std::string _input_larmatch_tree_name;       ///< name of tree with larmatch points [default: larmatch]
    std::string _input_adc_tree_name;            ///< name of tree with wire plane images [default: wire]
    std::string _input_taggerimg_tree_name;      ///< name of tree with wire-cell cosmic-tagged pixels [default: thrumu]
    std::string _ssnet_stem_name;                ///< stem name of trees with ssnet scores [default: ubspurn_plane]
    std::string _output_keypoint_tree_name;      ///< name of tree to output passing keypoints [default: taggerfilterkeypoint]
    std::string _output_filteredhits_tree_name;  ///< name of tree to output larflow3dhits that pass [default: taggerfilterhit]
    std::string _output_rejectedhits_tree_name;  ///< name of tree to output larflow3dhits that fail [default: taggerrejectedhit]
    bool        _save_rejected_hits;             ///< if true, save hits that fail the filter [default: false]

    /** @brief set default values for parameters */
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

    /** @brief set name of tree to get keypoints [default: taggerfilterkeypoint]*/
    void set_input_keypoint_tree_name( std::string keypoint )  { _input_keypoint_tree_name=keypoint; };

    /** @brief set name of tree to get larmatch space points [default: larmatch]*/
    void set_input_larmatch_tree_name( std::string larmatch )  { _input_larmatch_tree_name=larmatch; };

    /** @brief set name of tree to get wirecell cosmic-tagged image [default thrumu] */
    void set_input_taggerimg_tree_name( std::string tagger )   { _input_taggerimg_tree_name=tagger; };

    /** @brief set stem name of tree to get sparse ssnet images [default ubspurn_plane] */
    void set_input_ssnet_stem_name( std::string stem )         { _ssnet_stem_name=stem; };

    /** @brief set output keypoint tree name */
    void set_output_keypoint_tree_name( std::string keypoint ) { _output_keypoint_tree_name=keypoint; };

    /** @brief set passing  hits tree name */
    void set_output_filteredhits_tree_name( std::string hits ) { _output_filteredhits_tree_name=hits; };

    /** @brief set rejected hits tree name */
    void set_output_rejectedhits_tree_name( std::string hits ) { _output_rejectedhits_tree_name=hits; };

    /** @brief if set to true, save the rejected hits [default false] */
    void set_save_rejected_hits( bool save )                   { _save_rejected_hits=save; };
    
  };

}
}

#endif
