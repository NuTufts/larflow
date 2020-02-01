#ifndef __LARFLOW_PREP_PREPLARMATCHDATA_H__
#define __LARFLOW_PREP_PREPLARMATCHDATA_H__

#include <vector>
#include "larcv/core/DataFormat/Image2D.h"
#include "larflow/LArFlowConstants/LArFlowConstants.h"
#include "FlowTriples.h"
#include "TH2D.h"

namespace larflow {

  class PrepMatchTriplets {
  public:

    PrepMatchTriplets() {};
    virtual ~PrepMatchTriplets() {};

    void process( const std::vector<larcv::Image2D>& adc_v,
                  const std::vector<larcv::Image2D>& badch_v,
                  const float adc_threshold );

    std::vector<TH2D> plot_sparse_images( const std::vector<larcv::Image2D>& adc_v,
                                          std::string hist_stem_name );

    
    std::vector< std::vector< FlowTriples::PixData_t > >  _sparseimg_vv;    
    std::vector< std::vector<int> >                       _triplet_v;
    
  };

}

#endif
