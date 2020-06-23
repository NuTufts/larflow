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

    void split( const std::vector<larcv::Image2D>& ssnet_score_v,
                const larlite::event_larflow3dhit& lfhit_v,
                const float ssnet_score_threshold,
                const float larmatch_score_threshold,
                std::vector<larlite::larflow3dhit>& accept_v,
                std::vector<larlite::larflow3dhit>& reject_v );

    void process( larcv::IOManager& iolcv, larlite::storage_manager& ioll );

  protected:
    
    // params
    float _score_threshold;
    float _larmatch_threshold;
    std::string _ssnet_stem_name;
    std::string _adc_name;
    std::string _input_larmatch_hit_tree_name;
    std::string _output_larmatch_hit_stem_name;
    
  public:
    
    void set_ssnet_threshold( float thresh )    { _score_threshold=thresh; };
    void set_larmatch_threshold( float thresh ) { _larmatch_threshold=thresh; };
    void set_ssnet_tree_stem_name( std::string stem ) { _ssnet_stem_name=stem; };
    void set_larmatch_tree_name( std::string hitname ) { _input_larmatch_hit_tree_name=hitname; };
    void set_adc_tree_name( std::string name ) { _adc_name=name; };
    void set_output_tree_stem_name( std::string stem ) { _output_larmatch_hit_stem_name=stem; };

  protected:

    std::vector<larlite::larflow3dhit>  _shower_hit_v;
    std::vector<larlite::larflow3dhit>  _track_hit_v;

  public:

    std::vector<larlite::larflow3dhit>& get_shower_hits() { return _shower_hit_v; };
    std::vector<larlite::larflow3dhit>& get_track_hits()  { return _track_hit_v; };    

    const std::vector<larlite::larflow3dhit>& get_shower_hits() const { return _shower_hit_v; };
    const std::vector<larlite::larflow3dhit>& get_track_hits()  const { return _track_hit_v; };    
    
    
  };
  
}
}

#endif
