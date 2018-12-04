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

    typedef enum { kNNmax, kMaxDist } Neighborhood_t;
    
    CilantroSpectral( const std::vector<larlite::larflow3dhit>& larflowhits,
		      const CilantroSpectral::Neighborhood_t nn_type, // use neighrbors in radius, or max neighbors
		      const int NC,        // number of max clusters
		      const int MaxNN,     // used if kNNmax
		      const float MaxDist, // used if kMaxDist
		      const float Sigma,   // sigma used for weight kernel, if <0, we try to calculate it
		      bool debug=false );  // generate dummy points for testing
    virtual ~CilantroSpectral();

    void build_neighborhood_graph(std::vector<Eigen::Vector3f> &points, Eigen::SparseMatrix<float> &affinities);

    void get_cluster_indeces(std::vector<std::vector<long unsigned int> >& cpi, std::vector<long unsigned int>& idx_map);
    void generate_dummy_data(std::vector<Eigen::Vector3f>& points);

    // inputs
    const std::vector<larlite::larflow3dhit>* _larflowhits;
    Neighborhood_t _nn_type;
    int _NC;
    int _MaxNN;
    float _MaxDist;
    float _Sigma;

    
    std::vector<Eigen::Vector3f> _points; // or Eigen::Matrix<float,3,Eigen::Dynamic>
    Eigen::SparseMatrix<float>   _affinities;
    cilantro::SpectralClustering<float>* _sc;
    
    
  };

}

#endif
