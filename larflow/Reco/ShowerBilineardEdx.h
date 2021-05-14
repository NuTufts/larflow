#ifndef __LARFLOW_RECO_SHOWER_BILINEAR_DEDX_H__
#define __LARFLOW_RECO_SHOWER_BILINEAR_DEDX_H__

#include <vector>
#include "TGraph.h"
#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/Image2D.h"
#include "larcv/core/DataFormat/ImageMeta.h"
#include "DataFormat/larflowcluster.h"
#include "DataFormat/pcaxis.h"
#include "DataFormat/track.h"

namespace larflow {
namespace reco {

  class ShowerBilineardEdx : public larcv::larcv_base {
  public:
    
    ShowerBilineardEdx();
    virtual ~ShowerBilineardEdx();

    struct Result_t {
      std::vector<float> trunk_dedx_planes;
      std::vector< std::vector<float> > dedx_curve_planes;
    };

    void processShower( larlite::larflowcluster& shower,
                        larlite::track& trunk,
                        larlite::pcaxis& pca,
                        const std::vector<larcv::Image2D>& adc_v );

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
    
    std::vector<float> sumChargeAlongTrunk( const std::vector<float>& start3d,
                                            const std::vector<float>& end3d,
                                            const std::vector<larcv::Image2D>& img_v,
                                            const float threshold );
    

    // for debug
    std::vector< std::vector<TGraph> > bilinear_path_vv;
    
  };
  
}
}

#endif
