#include "CilantroSpectral.h"

namespace larflow {

  CilantroSpectral::CilantroSpectral( const std::vector<larlite::larflow3dhit>& larflowhits, const int NC, const int NN ) {
    _larflowhits = &larflowhits;
    // transfer points
    _points.clear();
    /*
    for ( auto const& hit : larflowhits ) {
      _points.push_back( Eigen::Vector3f(hit[0],hit[1],hit[2]) );
    }
    */
    generate_dummy_data(_points);
    std::cout << "num points: " << _points.size() << std::endl;
    build_neighborhood_graph(_points, _affinities, NN);

    _sc = new cilantro::SpectralClustering<float>(_affinities, NC, true, cilantro::GraphLaplacianType::NORMALIZED_RANDOM_WALK);
    std::cout << "Number of clusters: " << _sc->getNumberOfClusters() << std::endl;
    std::cout << "Performed k-means iterations: " << _sc->getClusterer().getNumberOfPerformedIterations() << std::endl;

  }

  CilantroSpectral::~CilantroSpectral() {
    delete _sc;
    _points.clear();
    // release allocated memory of matrix
    _affinities.resize(0,0);
    _affinities.data().squeeze();
  }

  void CilantroSpectral::generate_dummy_data(std::vector<Eigen::Vector3f>& points){
    points.resize(1700);
    for (size_t i = 0; i < 1500; i++) {
      points.at(i).setRandom().normalize();
    }
    for (size_t i = 1500; i < 1700; i++) {
      points.at(i).setRandom().normalize();
      points.at(i) *= 0.3f;
    }
  }
  
  void CilantroSpectral::build_neighborhood_graph(std::vector<Eigen::Vector3f> &points, Eigen::SparseMatrix<float> &affinities, const int NN){
    std::cout << "num neighbors: " << NN << std::endl;
    // NN number of neighbors
    auto nh = cilantro::kNNNeighborhood<float>(NN);
    std::vector<cilantro::NeighborSet<float>> nn;
    cilantro::KDTree3f(points).search(points, nh, nn);
    std::cout << "populated kd tree" << std::endl;
    affinities = cilantro::getNNGraphFunctionValueSparseMatrix(nn, cilantro::RBFKernelWeightEvaluator<float>(), true);
    std::cout << "built affinity matrix" << std::endl;
  }
  
  void CilantroSpectral::get_cluster_indeces(cilantro::SpectralClustering<float>* sc,std::vector<std::vector<long unsigned int> >& cpi, std::vector<long unsigned int>& idx_map){
    cpi = sc->getClusterPointIndices();
    idx_map = sc->getClusterIndexMap();

  }

}
