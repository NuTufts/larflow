#include "CilantroSpectral.h"

namespace larflow {

  CilantroSpectral::CilantroSpectral( const std::vector<larlite::larflow3dhit>& larflowhits, const int NC, const int NN ) {
    _larflowhits = &larflowhits;
    // transfer points
    _points.clear();
    for ( auto const& hit : larflowhits ) {
      _points.push_back( Eigen::Vector3f(hit[0],hit[1],hit[2]) );
    }
    
    build_neighborhood_graph(_points, _affinities, NN);
    cilantro::Timer timer;
    timer.start();
    
    _sc = new cilantro::SpectralClustering<float>(_affinities, NC, true, cilantro::GraphLaplacianType::NORMALIZED_RANDOM_WALK);
    timer.stop();
    std::cout << "Clustering time: " << timer.getElapsedTime() << "ms" << std::endl;
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

  void CilantroSpectral::build_neighborhood_graph(std::vector<Eigen::Vector3f> &points, Eigen::SparseMatrix<float> &affinities, const int NN){
    // NN neighbors
    auto nh = cilantro::kNNNeighborhood<float>(NN);
    std::vector<cilantro::NeighborSet<float>> nn;
    cilantro::KDTree3f(points).search(points, nh, nn);
    affinities = cilantro::getNNGraphFunctionValueSparseMatrix(nn, cilantro::RBFKernelWeightEvaluator<float>(), true);
    
  }
  
  void CilantroSpectral::get_cluster_indeces(){
    const auto& cpi = _sc->getClusterPointIndices();
    size_t mins = _points.size(), maxs = 0;
    for (size_t i = 0; i < cpi.size(); i++) {
      if (cpi[i].size() < mins) mins = cpi[i].size();
      if (cpi[i].size() > maxs) maxs = cpi[i].size();
    }
    std::cout << "Cluster size range is: [" << mins << "," << maxs << "]" << std::endl;

    const auto& idx_map = _sc->getClusterIndexMap();
    for(int i=0; i<idx_map.size(); i++){
      std::cout << "cluster id "<< idx_map.at(i) << std::endl;
    }
    
  }

}
