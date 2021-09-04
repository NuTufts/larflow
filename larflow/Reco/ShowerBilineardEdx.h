#ifndef __LARFLOW_RECO_SHOWER_BILINEAR_DEDX_H__
#define __LARFLOW_RECO_SHOWER_BILINEAR_DEDX_H__

#include <vector>
#include <map>
#include "TGraph.h"
#include "TTree.h"
#include "TH2D.h"
#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/Image2D.h"
#include "larcv/core/DataFormat/ImageMeta.h"
#include "larlite/DataFormat/larflowcluster.h"
#include "larlite/DataFormat/pcaxis.h"
#include "larlite/DataFormat/track.h"
#include "larlite/DataFormat/mcshower.h"
#include "larlite/DataFormat/mctrack.h"
#include "larlite/LArUtil/SpaceChargeMicroBooNE.h"
#include "NuVertexCandidate.h"

namespace larflow {
namespace reco {

  /**
   * @class ShowerBilineardEdx
   * @ingroup Reco
   *
   * Use bilinear interpolation to get pixel value along track.
   * Not used. But code left here in case it's useful in the future.
   *
   */
  class ShowerBilineardEdx : public larcv::larcv_base {
  public:
    
    ShowerBilineardEdx();
    virtual ~ShowerBilineardEdx();

    struct Result_t {
      std::vector<float> trunk_dedx_planes;
      std::vector< std::vector<float> > dedx_curve_planes;
    };

    float aveBilinearCharge_with_grad( const larcv::Image2D& img,
                                       std::vector<float>& start3d,
                                       std::vector<float>& end3d,
                                       int npts,
                                       float avedQdx,
                                       std::vector<float>& grad );

    float colcoordinate_and_grad( const std::vector<float>& pos,
                                  const int plane,
                                  const larcv::ImageMeta& meta,
                                  std::vector<float>& grad );
    
    float rowcoordinate_and_grad( const std::vector<float>& pos,
                                  const larcv::ImageMeta& meta,
                                  std::vector<float>& grad );

    float bilinearPixelValue_and_grad( std::vector<float>& pos3d,
                                       const int plane,
                                       const larcv::Image2D& img,
                                       std::vector<float>& grad );
    
    std::vector< std::vector<TGraph> > bilinear_path_vv;
    
  private:
    
    static int ndebugcount;
    static larutil::SpaceChargeMicroBooNE* _psce;
    
  };
  
}
}

#endif
