#ifndef __SPLIT_HITS_BY_SSNET_H__
#define __SPLIT_HITS_BY_SSNET_H__

#include <string>
#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/Image2D.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "DataFormat/storage_manager.h"
#include "DataFormat/larflow3dhit.h"

namespace larflow {
namespace reco {

  /**
   * @ingroup Reco
   * @class SplitHitsBySSNet
   * @brief Separate a set of larflow3dhit using the SSNet score
   *
   */
  class SplitHitsBySSNet : public larcv::larcv_base {

  public:

    SplitHitsBySSNet()
      : larcv::larcv_base("SplitHitsBySSNet"),
      _score_threshold(0.5),
      _larmatch_threshold(0.1),
      _ssnet_stem_name("ubspurn_plane"),
      _adc_name("wire"),
      _input_larmatch_hit_tree_name("larmatch"),
      _output_larmatch_hit_stem_name("ssnsetsplit")
      {};
    virtual ~SplitHitsBySSNet() {};

    void label_and_split( const std::vector<larcv::Image2D>& ssnet_score_v,
                          const larlite::event_larflow3dhit& lfhit_v,
                          const float ssnet_score_threshold,
                          const float larmatch_score_threshold,
                          std::vector<larlite::larflow3dhit>& accept_v,
                          std::vector<larlite::larflow3dhit>& reject_v );

    void label( const std::vector<larcv::Image2D>& ssnet_score_v,
                larlite::event_larflow3dhit& larmatch_hit_v );

    void split( larlite::event_larflow3dhit& lfhit_v,
                const float ssnet_score_threshold,
                const float larmatch_score_threshold,
                std::vector<larlite::larflow3dhit>& accept_v,
                std::vector<larlite::larflow3dhit>& reject_v );

    void split_constinput( const larlite::event_larflow3dhit& lfhit_v,
                           const float ssnet_score_threshold,
                           const float larmatch_score_threshold,
                           std::vector<larlite::larflow3dhit>& accept_v,
                           std::vector<larlite::larflow3dhit>& reject_v );
    
    void process( larcv::IOManager& iolcv, larlite::storage_manager& ioll );
    
    void process_labelonly( larcv::IOManager& iolcv, larlite::storage_manager& ioll );
    void process_splitonly( larcv::IOManager& iolcv, larlite::storage_manager& ioll );        
    
  protected:
    
    // params
    float _score_threshold;                      ///< ssnet shower score threshold to be labeled shower
    float _larmatch_threshold;                   ///< larmatch score threshold to be considered
    std::string _ssnet_stem_name;                ///< stem name of tree holding ssnet images
    std::string _adc_name;                       ///< name of tree holding wire plane images
    std::string _input_larmatch_hit_tree_name;   ///< name of tree holding larflow3dhit made by larmatch network
    std::string _output_larmatch_hit_stem_name;  ///< stem name of tree holding output hits
    
  public:

    /** @brief set the ssnet shower score threshold */
    void set_ssnet_threshold( float thresh )    { _score_threshold=thresh; };

    /** @brief set the larmatch treshold score */
    void set_larmatch_threshold( float thresh ) { _larmatch_threshold=thresh; };

    /** @brief set stem name of tree holding ssnet images */
    void set_ssnet_tree_stem_name( std::string stem ) { _ssnet_stem_name=stem; };

    /** @brief set name of tree holding larmatch hits */
    void set_larmatch_tree_name( std::string hitname ) { _input_larmatch_hit_tree_name=hitname; };

    /** @brief set name of tree holding wire plane images */
    void set_adc_tree_name( std::string name ) { _adc_name=name; };

    /** @brief set name of tree to put output hits */
    void set_output_tree_stem_name( std::string stem ) { _output_larmatch_hit_stem_name=stem; };

  protected:

    std::vector<larlite::larflow3dhit>  _shower_hit_v; ///< container holding shower-labeled hits
    std::vector<larlite::larflow3dhit>  _track_hit_v;  ///< container holding track-labeled hits

  public:

    /** @brief get mutable shower larflow3dhit container */
    std::vector<larlite::larflow3dhit>& get_shower_hits() { return _shower_hit_v; };

    /** @brief get mutable track larflow3dhit container */    
    std::vector<larlite::larflow3dhit>& get_track_hits()  { return _track_hit_v; };    

    /** @brief get const shower larflow3dhit container */    
    const std::vector<larlite::larflow3dhit>& get_shower_hits() const { return _shower_hit_v; };

    /** @brief get const track larflow3dhit container */        
    const std::vector<larlite::larflow3dhit>& get_track_hits()  const { return _track_hit_v; };    
    
    
  };
  
}
}

#endif
