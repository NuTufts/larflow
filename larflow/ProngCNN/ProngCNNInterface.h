#ifndef __LARFLOW_PRONG_CNN_INTERFACE_H__
#define __LARFLOW_PRONG_CNN_INTERFACE_H__

#include <vector>
#include "TVector3.h"
#include "larlite/DataFormat/larflowcluster.h"
#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/Image2D.h"

#ifndef __CINT__
#ifndef __CLING__

// hide torch stuff from the interpretter
#include "larpid/model/TorchModel.h"
// //Load ROOT ClassDef back in now that we have the torch headers:
// #ifdef ClassDef
// #undef ClassDef
// #endif
// #include <Rtypes.h>

#endif
#endif

#include "larcv/core/DataFormat/IOManager.h"

namespace larflow {
namespace prongcnn {


  class ProngCNNInterface {
  //}: public larcv::larcv_base {

  public:
    
    ProngCNNInterface()
      : //larcv::larcv_base("ProngCNNInterface"),
      _model_loaded(false),
      fPixelThreshold(10)
      {};
    ~ProngCNNInterface() {};

    bool load_model( std::string model_file, bool fdebug=false );

    bool get_larpid_prong_scores( const TVector3& cropPt,
        const larlite::larflowcluster& hitcluster,
        larcv::IOManager& iolcv,
        bool preserve_shower_pixels,
        std::vector<float>& pid_scores,
        std::vector<float>& primary_and_parent_scores,
        int& process,
        float& purity_score,
        float& completeness_score,
        int& nplanes_above );
    
  protected:
    
    bool _model_loaded;
    int fPixelThreshold;

#ifndef __CINT__
#ifndef __CLING__
    // Hide torch stuff from the ROOT interpretter
    larpid::model::TorchModel _model;

#endif
#endif
    
  };
  
}
}

#endif
