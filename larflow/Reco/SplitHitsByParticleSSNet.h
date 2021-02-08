#ifndef __SPLIT_HITS_BY_PARTICLE_SSNET_H__
#define __SPLIT_HITS_BY_PARTICLE_SSNET_H__

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
   * @class SplitHitsByParticleSSNet
   * @brief Separate a set of larflow3dhit using the SSNet score
   *
   */
  class SplitHitsByParticleSSNet : public larcv::larcv_base {

  public:

    SplitHitsByParticleSSNet()
      : larcv::larcv_base("SplitHitsByParticleSSNet"),
      _score_threshold(0.5),
      _larmatch_threshold(0.1),
      _adc_name("wire"),
      _input_larmatch_hit_tree_name("larmatch"),
      _input_ssnet_tree_name("sparseuresnetout"),
      _output_larmatch_hit_stem_name("fiveparticlessn")
      {};
    virtual ~SplitHitsByParticleSSNet() {};

    void process( larcv::IOManager& iolcv, larlite::storage_manager& ioll );

  protected:
    
    // params
    float _score_threshold;                      ///< ssnet shower score threshold to be labeled shower
    float _larmatch_threshold;                   ///< larmatch score threshold to be considered
    std::string _adc_name;                       ///< name of tree holding wire plane images
    std::string _input_larmatch_hit_tree_name;   ///< name of tree holding larflow3dhit made by larmatch network
    std::string _input_ssnet_tree_name;          ///< name of tree holding sparse image ssnet output
    std::string _output_larmatch_hit_stem_name;  ///< stem name of tree holding output hits
    
  public:


  protected:

    std::vector<larlite::larflow3dhit>  _mip_hit_v; ///< container holding shower-labeled hits
    std::vector<larlite::larflow3dhit>  _hip_hit_v; ///< container holding shower-labeled hits
    std::vector<larlite::larflow3dhit>  _shower_hit_v; ///< container holding gamma/electron hits
    std::vector<larlite::larflow3dhit>  _michel_hit_v; ///< container holding michel electron hits            
    std::vector<larlite::larflow3dhit>  _delta_hit_v;  ///< container holding delta electron hits

  };
  
}
}

#endif
