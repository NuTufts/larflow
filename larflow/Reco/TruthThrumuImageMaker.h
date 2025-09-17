#ifndef __LARFLOW_RECO_TRUTH_THRUMU_IMAGEMAKER_H__
#define __LARFLOW_RECO_TRUTH_THRUMU_IMAGEMAKER_H__


#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "larlite/DataFormat/storage_manager.h"

#include <string>

namespace larflow {
namespace reco {

  class TruthThrumuImageMaker : public larcv::larcv_base {

    public:

    TruthThrumuImageMaker()
    : larcv::larcv_base("TruthThrumuImageMaker"),
    output_thrumu_treename("thrumu"),
    input_image_treename("wire")
    {};

    ~TruthThrumuImageMaker() {};

    void process( larcv::IOManager& iolcv, larlite::storage_manager& ioll );


    std::string output_thrumu_treename;
    std::string input_image_treename;

  };

}
}

#endif
