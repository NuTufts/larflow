#include "CilantroPCA.h"

namespace larflow {

  CilantroPCA::CilantroPCA( const larlite::larflowcluster& cluster ) {
    _cluster = &cluster;
    // transfer points
    _points.clear();
    for ( auto const& hit : cluster ) {
      bool hitok = true;
      for ( int i=0; i<3; i++) {
	if ( std::isnan(hit[i]) ) hitok = false;
      }
      _points.push_back( Eigen::Vector3f(hit[0],hit[1],hit[2]) );
    }
    _pca = new cilantro::PrincipalComponentAnalysis3f(_points);

    // std::cout << "[CilantroPCA::CilantroPCA] Data mean: " << _pca->getDataMean().transpose() << std::endl;
    // std::cout << "[CilantroPCA::CilantroPCA] Eigenvalues: " << _pca->getEigenValues().transpose() << std::endl;
    // std::cout << "[CilantroPCA::CilantroPCA] Eigenvectors: " << std::endl << _pca->getEigenVectors() << std::endl;
  }

  CilantroPCA::~CilantroPCA() {
    delete _pca;
    _points.clear();
  }

  larlite::pcaxis CilantroPCA::getpcaxis() {
    // pcaxis::pcaxis(bool                               ok,
    // 		 int                                nHits,
    // 		 const double*                      eigenValues,
    // 		 const larlite::pcaxis::EigenVectors& eigenVecs,
    // 		 const double*                      avePos,
    // 		 const double                       aveHitDoca,
    // 		 size_t                             id) :
    
    double eigenvalues[3];
    double avePos[3];
    larlite::pcaxis::EigenVectors eigenVecs(3);
    for (int i=0; i<3; i++) {
      avePos[i]      = _pca->getDataMean()(i);
      eigenvalues[i] = _pca->getEigenValues()(i);
      eigenVecs[i].resize(3,0);
      for (int j=0; j<3; j++)
	eigenVecs[i][j] = _pca->getEigenVectors()(i,j);
    }
    
    larlite::pcaxis axis( true, _points.size(), eigenvalues, eigenVecs, avePos, 0.0, 0);
    return axis;
  }
  
  

}
