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

    void make_truth_vector( const std::vector<larcv::Image2D>& larflow_v );

    
    std::vector< std::vector< FlowTriples::PixData_t > >  _sparseimg_vv;   ///< sparse representation of image
    std::vector< std::vector<int> >                       _triplet_v;      ///< set of sparseimage indices indicating candidate 3-plane match
    std::vector< int >                                    _truth_v;        ///< indicates if index set in _triple_v is true match (1) or not (0)
    std::vector< float >                                  _weight_v;       ///< assigned weight for triplet
    std::vector< larflow::FlowDir_t >                     _flowdir_v;      ///< flow direction te triplet comes from

    
  };

}

#endif
