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

    CilantroSpectral( const std::vector<larlite::larflow3dhit>& larflowhits, const int NC, const int NN );
    virtual ~CilantroSpectral();

    void build_neighborhood_graph(std::vector<Eigen::Vector3f> &points, Eigen::SparseMatrix<float> &affinities, const int NN);

    void get_cluster_indeces(std::vector<std::vector<long unsigned int> >& cpi, std::vector<long unsigned int>& idx_map);
    void generate_dummy_data(std::vector<Eigen::Vector3f>& points);
    
    const std::vector<larlite::larflow3dhit>* _larflowhits;
    std::vector<Eigen::Vector3f> _points; // or Eigen::Matrix<float,3,Eigen::Dynamic>
    Eigen::SparseMatrix<float>   _affinities;
    cilantro::SpectralClustering<float>* _sc;
    
    
  };

}

#endif
