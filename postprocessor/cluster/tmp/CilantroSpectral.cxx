#include "CilantroSpectral.h"

namespace larflow {

  CilantroSpectral::CilantroSpectral( const std::vector<larlite::larflow3dhit>& larflowhits, // input larflow hits
				      const CilantroSpectral::Neighborhood_t nn_type, // use neighrbors in radius, or max neighbors
				      const int NC,        // number of max clusters
				      const int MaxNN,     // used if kNNmax
				      const float MaxDist, // used if kMaxDist
				      const float Sigma,   // sigma used for weight kernel, if <0, we try to calculate it
				      bool debug )  // generate dummy points for testing
    : _larflowhits( &larflowhits ),
      _nn_type( nn_type ),
      _NC( NC ),
      _MaxNN( MaxNN ),
      _MaxDist( MaxDist ),
      _Sigma( Sigma )
  {
    _larflowhits = &larflowhits;
    // transfer points
    _points.clear();
    
    if(!debug){
      for ( auto const& hit : larflowhits ) {
	_points.push_back( Eigen::Vector3f(hit[0],hit[1],hit[2]) );
      }
    }
    else{
      generate_dummy_data(_points);
    }
    std::cout << "num points: " << _points.size() << std::endl;
    build_neighborhood_graph(_points, _affinities);

    _sc = new cilantro::SpectralClustering<float>(_affinities, NC, true, cilantro::GraphLaplacianType::NORMALIZED_RANDOM_WALK, 200,1.e-7,false);
    //_sc = new cilantro::SpectralClustering<float>(_affinities, NC, true, cilantro::GraphLaplacianType::UNNORMALIZED,200,1.e-7,false);
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
  
  void CilantroSpectral::build_neighborhood_graph(std::vector<Eigen::Vector3f> &points, Eigen::SparseMatrix<float> &affinities){

    if ( _nn_type==kNNmax )
      std::cout << "[CilantroSpectral::build_neighborhood_graph][INFO] Making affinities using max NN of " << _MaxNN << std::endl;
    else
      std::cout << "[CilantroSpectral::build_neighborhood_graph][INFO] Making affinities using NN within radius of " << _MaxDist << std::endl;

    bool gensig = ( _Sigma<0 ) ? true : false;
    if ( gensig ) {
      std::cout << "[CilantroSpectral::build_neighborhood_graph][INFO] Generating sigma, the length scale between hits." << std::endl;
    }

    std::vector<cilantro::NeighborSet<float>> nn;
    
    if ( _nn_type==kNNmax ) {      
      // NN number of neighbors
      auto nh = cilantro::kNNNeighborhood<float>(_MaxNN);
      std::vector<cilantro::NeighborSet<float>> nn_nnmax;
      cilantro::KDTree3f(points).search(points, nh, nn_nnmax);
      std::cout << "[CilantroSpectral::build_neighborhood_graph][INFO] populated kd tree (with max NN)" << std::endl;
      std::swap(nn_nnmax,nn); // pass vector out of scope
    }
    else if ( _nn_type==kMaxDist ) {
      auto nh = cilantro::radiusNeighborhood<float>(_MaxDist);
      std::vector< cilantro::NeighborSet<float> > nn_maxdist;
      cilantro::KDTree3f(points).search(points,nh,nn_maxdist);
      std::cout << "[CilantroSpectral::build_neighborhood_graph][INFO] populated kd tree (within max dist)" << std::endl;
      std::swap(nn_maxdist,nn); // pass vector out of scope
    }

    if ( gensig ) {
      int npoints = 0.;
      _Sigma = 0.;
      for ( auto const& nnset : nn ) {
	for ( auto const& neighbor : nnset ) {
	  _Sigma += sqrt(neighbor.value); // look at kd_tree.hpp definition of KDTree3f: distances are squared
	  npoints++;
	}
      }
      if ( npoints>0 )
	_Sigma /= float(npoints);
      std::cout << "[CilantroSpectral::build_neighborhood_graph][INFO] calculated mean neighbor distance of " << _Sigma << " (using " << npoints << " neighbors)" << std::endl;
    }


    // build affinity matrix
    bool dist_squared = true;
    affinities = cilantro::getNNGraphFunctionValueSparseMatrix(nn, cilantro::RBFKernelWeightEvaluator<float>( _Sigma ), dist_squared);
    std::cout << "[CilantroSpectral::build_neighborhood_graph][INFO] built affinity matrix" << std::endl;

  }
  
  void CilantroSpectral::get_cluster_indeces(std::vector<std::vector<long unsigned int> >& cpi, std::vector<long unsigned int>& idx_map){
    cpi     = _sc->getClusterPointIndices();
    idx_map = _sc->getClusterIndexMap();
  }

}
