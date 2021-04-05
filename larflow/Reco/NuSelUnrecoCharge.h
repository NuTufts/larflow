#ifndef __NUSEL_UNRECO_CHARGE_H__
#define __NUSEL_UNRECO_CHARGE_H__

#include <vector>
#include "TTree.h"

#include "DataFormat/storage_manager.h"
#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"

#include "NuVertexCandidate.h"
#include "NuSelectionVariables.h"

namespace larflow {
namespace reco {

  /**
   * @ingroup Reco
   * @class NuSelUnrecoCharge
   * @brief Provides measure for the amount of unreconstructed charge around the
   *        neutrino candidate. Helps remove higher energy BNB-nu events 
   *        for Gen-2 nu-e selection.
   */
  
  class NuSelUnrecoCharge : public larcv::larcv_base {

  public:

    NuSelUnrecoCharge()
      : larcv::larcv_base("NuSelUnrecoCharge"),
      _ksave_mask(false),
      _tree(nullptr)
      {};
    virtual ~NuSelUnrecoCharge() {};

    void analyze( larcv::IOManager& iolcv,
                  larlite::storage_manager& ioll,
                  larflow::reco::NuVertexCandidate& nuvtx,
                  larflow::reco::NuSelectionVariables& output );

    void bindVarsToTree( TTree* tree );
    void clearVars();
    void fillTree() { if ( _tree ) { _tree->Fill(); } };
    void setSaveMask( bool save ) { _ksave_mask=save; };

  protected:

    void _count_unreco_pixels( std::vector<larcv::Image2D>& numask_v,
                               const std::vector<larcv::Image2D>& adc_v,
                               const std::vector<larcv::Image2D>& thrumu_v,
                               const float adc_threshold,
			       std::vector<int>& unreco_intime_counts,			       
                               std::vector<int>& unreco_counts,
                               std::vector<float>& unreco_fraction );

    bool _ksave_mask;

    TTree* _tree;
    std::vector<int>   _intime_count_v;
    std::vector<int>   _unreco_count_v;
    std::vector<float> _unreco_fraction_v;
    float _median_fraction;
    float _min_fraction;
    float _max_fraction;

  };
  
}
}

#endif
