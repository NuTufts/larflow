#ifndef __CILANTRO_PCA_H__
#define __CILANTRO_PCA_H__

#include <vector>
#include <cilantro/principal_component_analysis.hpp>
#include "DataFormat/larflowcluster.h"
#include "DataFormat/pcaxis.h"

namespace larflow {

  class CilantroPCA {
  public:

    CilantroPCA( const larlite::larflowcluster& cluster );
    virtual ~CilantroPCA();

    larlite::pcaxis getpcaxis();
    
    const larlite::larflowcluster* _cluster;
    std::vector<Eigen::Vector3f>   _points;
    cilantro::PrincipalComponentAnalysis3f* _pca;

    
    
  };

}

#endif
