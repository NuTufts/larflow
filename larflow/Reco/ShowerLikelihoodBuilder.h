#ifndef __LARFLOW_RECO_SHOWERLIKELIHOODBUILDER_H__
#define __LARFLOW_RECO_SHOWERLIKELIHOODBUILDER_H__

/**
 * This class contains functions to build distibutions needed 
 * to make shower likelihood functions to be used in the larfow shower reco code
 *
 * This includes:
 *
 *  1) shower profile likelihood in 3D. its the location of charge deposited as a function of 
 *       the distance along the trunk line and the perpendicular dist from the trunk line
 *  2) brem segment impact param, distance to trunk line, cosine of pca between trunk lines
 *
 * We fill a tree to later use to make distributions.
 * Can feed it, single shower MC  (best) or low energy neutrino. OK.
 *
 */

#include "larcv/core/DataFormat/IOManager.h"
#include "DataFormat/storage_manager.h"
#include "DataFormat/larflow3dhit.h"
#include "LArUtil/SpaceChargeMicroBooNE.h"

#include "larflow/PrepFlowMatchData/PrepMatchTriplets.h"

#include "TFile.h"
#include "TH2F.h"

namespace larflow {
namespace reco {

  class ShowerLikelihoodBuilder {

  public:

    ShowerLikelihoodBuilder();
    virtual ~ShowerLikelihoodBuilder();

    void process( larcv::IOManager& iolcv, larlite::storage_manager& ioll );

    TH2F* _hll;
    larflow::PrepMatchTriplets tripletalgo;

    void _fillProfileHist( const std::vector<larlite::larflow3dhit>& truehit_v,
                           std::vector<float>& shower_dir,
                           std::vector<float>& shower_vtx );

    larutil::SpaceChargeMicroBooNE* _psce;
    
  };
  
}
}

#endif
