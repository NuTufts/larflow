#ifndef __LARFLOW_RECO_NUVERTEX_PRONGCNN_H__
#define __LARFLOW_RECO_NUVERTEX_PRONGCNN_H__

/**
 * @brief This class provides an interface to the ProngCNN C++ interface for NuVertexCandidate objects
 *
 */

#include "larcv/core/DataFormat/IOManager.h"
#include "larflow/Reco/NuVertexCandidate.h"

namespace larflow {
namespace reco {

  class NuVertexProngCNN : public larcv::larcv_base {

  public:

    NuVertexProngCNN();
    ~NuVertexProngCNN();

    void runProngCNN( larflow::reco::NuVertexCandidate& nuvtx,
		      larcv::IOManager& ioman,
		      bool keep_cosmic_shower_pixels );
    
    
  };
  
}
}

#endif
