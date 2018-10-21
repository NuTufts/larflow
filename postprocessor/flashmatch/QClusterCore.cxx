#include "QClusterCore.h"

// larlite
#include "LArUtil/Geometry.h"
#include "LArUtil/LArProperties.h"
#include "LArUtil/SpaceChargeMicroBooNE.h"
#include "LArUtil/TimeService.h"
#include "SelectionTool/OpT0Finder/PhotonLibrary/PhotonVisibilityService.h"

// cluster
#include "cluster/CoreFilter.h"
#include "cluster/CilantroPCA.h"

namespace larflow {

    QClusterCore::QClusterCore( const QCluster_t& qcluster ) :
    _cluster(&qcluster)
  {
    buildCore();

  }

  void QClusterCore::buildCore() {
    
    int minneighbors = 3;
    int minclusterpoints = 5;
    float maxdist = 10.0;

    // const larutil::Geometry* geo = larutil::Geometry::GetME();
    // const larutil::LArProperties* larp = larutil::LArProperties::GetME();    
    // const float  driftv = larp->DriftVelocity();    

    std::vector< std::vector<float> > clusterpts;
    clusterpts.reserve(_cluster->size()); // need to define a non-copy way to do this ...
    for ( auto const& qhit : *_cluster ) {
      std::vector<float> qpt = qhit.xyz;
      // substract the flash time
      // float xoffset = (_flashdata->tpc_tick-3200)*0.5*driftv;
      // qpt[0] -= xoffset;
      clusterpts.push_back( qhit.xyz );
    }

    // core filter
    CoreFilter corealgo( clusterpts, minneighbors, maxdist );

    // we take a guess that the largest cluster is the core
    int largest_idx = corealgo.getIndexOfLargestCluster();
    const std::vector<int>& core_indices = corealgo.getClusterIndices().at(largest_idx);
    std::vector< std::vector<float> > core(core_indices.size());
    int icore=0;
    for ( auto const& idx : core_indices ) {
      core[icore] = clusterpts[idx];
    }

    // we calculate the core's pca
    CilantroPCA corepc_algo( core );
    larlite::pcaxis corepca = corepc_algo.getpcaxis();

    // the corepca defines a line. all other subclusters must be within some distance from this line
    Eigen::Vector3f origin( corepca.getAvePosition()[0], corepca.getAvePosition()[1], corepca.getAvePosition()[2] );
    Eigen::Vector3f vec( corepca.getEigenVectors()[0][0], corepca.getEigenVectors()[1][0], corepca.getEigenVectors()[2][0] );
    Eigen::ParametrizedLine< float, 3 > coreline( origin, vec );

    // loose addition, just one points need to be close
    std::set< int > joinlist;
    int joinsize = core_indices.size();
    for ( int iclust=0; iclust<(int)corealgo.getClusterIndices().size(); iclust++ ) {

      if ( iclust==largest_idx ) continue;

      bool join2core = false;
      for (auto& idx : corealgo.getClusterIndices().at(iclust) ) {
	Eigen::Map< Eigen::Vector3f > testpt( clusterpts[idx].data() );
	float dist = coreline.distance(testpt); // eigen is great
	if ( dist < 10.0 ) {
	  join2core = true;
	  break;
	}
      }

      if (join2core ) {
	joinlist.insert( iclust );
	joinsize += corealgo.getClusterIndices().at(iclust).size();
      }
    }

    // make core, make non-core points
    _core.clear();
    _core.reserve( joinsize );
    for (auto& idx : core_indices ) {
      _core.push_back( (*_cluster)[idx] );
    }
    for ( auto& iclust : joinlist )  {
      for (auto& idx : corealgo.getClusterIndices().at(iclust) )
	_core.push_back( (*_cluster)[idx] );
    }
    _core.idx = (*_cluster).idx;
    _core.mctrackid = (*_cluster).mctrackid;
    _core.truthmatched_flashidx = (*_cluster).truthmatched_flashidx;

    // not core
    for ( int iclust=0; iclust<iclust<(int)corealgo.getClusterIndices().size(); iclust++ ) {
      auto it = joinlist.find(iclust);
      if ( it!=joinlist.end() ) continue;

      QCluster_t qnoncore;
      for (auto& idx : corealgo.getClusterIndices().at(iclust) )
	qnoncore.push_back( (*_cluster)[idx] );
      qnoncore.idx = (*_cluster).idx;
      qnoncore.mctrackid = (*_cluster).mctrackid;
      qnoncore.truthmatched_flashidx = (*_cluster).truthmatched_flashidx;
      
      _noncore.emplace_back( std::move(qnoncore) );
    }

    // pca's
    //CilantroPCA corepc_algo( core );
    //larlite::pcaxis corepca = corepc_algo.getpcaxis();
    std::vector< std::vector<float> > finalcorepts(_core.size());
    for ( int ihit=0; ihit<(int)_core.size(); ihit++ ) {
      finalcorepts[ihit] = _core[ihit].xyz;
    }
    CilantroPCA finalcorepca( finalcorepts );
    _pca_core = finalcorepca.getpcaxis();

    // non-core pca
    for ( int inoncore=0; inoncore<(int)_noncore.size(); inoncore++ ) {
      QCluster_t& qnoncore = _noncore[inoncore];
      
      std::vector< std::vector<float> > noncorepts(qnoncore.size());
      for ( int ihit=0; ihit<(int)qnoncore.size(); ihit++ ) {
	noncorepts[ihit] = qnoncore[ihit].xyz;
      }
      CilantroPCA noncorepca( noncorepts );
      _pca_noncore.push_back( noncorepca.getpcaxis() );
    }//noncore loop

    
  }//end of define core

  void QClusterCore::fillGaps() {
    
  }

}
