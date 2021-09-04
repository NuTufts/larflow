#ifndef __LARFLOW_RECO_COSMIC_PROTON_FINDER_H__
#define __LARFLOW_RECO_COSMIC_PROTON_FINDER_H__

#include <vector>
#include <string>
#include "larlite/DataFormat/storage_manager.h"
#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "LikelihoodProtonMuon.h"

namespace larflow {
namespace reco {

  /**
   * @brief Analyze cosmic muon tracks using dqdx to find short protons and reclassify as in-time for nu-interaction building
   *
   */
  class CosmicProtonFinder : public larcv::larcv_base {

  public:

    CosmicProtonFinder();
    virtual ~CosmicProtonFinder() {};

    void process( larcv::IOManager& iolcv,
                  larlite::storage_manager& ioll );

    float get_length( const larlite::track& lltrack );

    larflow::reco::LikelihoodProtonMuon _llpid;

    std::vector<std::string> _input_cosmic_treename_v;
    std::string _output_tree_name;
    
  };
  
}
}

#endif
