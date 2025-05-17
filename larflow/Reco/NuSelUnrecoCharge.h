#ifndef __NUSEL_UNRECO_CHARGE_H__
#define __NUSEL_UNRECO_CHARGE_H__

#include <vector>
#include "TTree.h"

#include "larlite/DataFormat/storage_manager.h"
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

    void analyze_with_spacepoints( larcv::IOManager& iolcv,
				   larlite::storage_manager& ioll,
				   larflow::reco::NuVertexCandidate& nuvtx,
				   larflow::reco::NuSelectionVariables& output );
    

  protected:

    void _count_unreco_pixels( std::vector<larcv::Image2D>& numask_v,
                               const std::vector<larcv::Image2D>& adc_v,
                               const std::vector<larcv::Image2D>& thrumu_v,
                               const float adc_threshold,
                               std::vector<int>& all_intime_counts,   // num above thresh pixels intime
                               std::vector<int>& reco_intime_counts,  // reco pixel intime
                               std::vector<int>& reco_outtime_counts, // reco pixel outtime
                               std::vector<int>& intime_unreco_counts, // intime but not recod						
                               std::vector<float>& unreco_fraction,
                               std::vector<float>& cosmic_reco_fraction  );

    bool _ksave_mask;

    TTree* _tree;
    std::vector<int>   _intime_count_v;
    std::vector<int>   _reco_outtime_count_v;
    std::vector<int>   _unreco_count_v;
    std::vector<int>   _reco_count_v;
    std::vector<float> _unreco_fraction_v;
    float _median_fraction;
    float _min_fraction;
    float _max_fraction;

  };
  
}
}

#endif
