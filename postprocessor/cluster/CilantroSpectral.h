#ifndef __CILANTRO_SPECTRAL_H__
#define __CILANTRO_SPECTRAL_H__

#include <vector>
#include <cilantro/nearest_neighbor_graph_utilities.hpp>
#include <cilantro/spectral_clustering.hpp>
#include <cilantro/timer.hpp>
#include "DataFormat/larflowcluster.h"

namespace larflow {

  class CilantroSpectral {
  public:

    CilantroSpectral( const larlite::larflowcluster& cluster, const int NC, const int NN );
    virtual ~CilantroSpectral();

    void build_neighborhood_graph(std::vector<Eigen::Vector3f> &points, Eigen::SparseMatrix<float> &affinities, const int NN);
    
    const larlite::larflowcluster* _cluster;
    std::vector<Eigen::Vector3f> _points; // or Eigen::Matrix<float,3,Eigen::Dynamic>
    Eigen::SparseMatrix<float>   _affinities;
    cilantro::SpectralClustering<float>* _sc;
    
    
  };

}

#endif
