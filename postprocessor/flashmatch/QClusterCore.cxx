#include "QClusterCore.h"

// larlite
#include "LArUtil/Geometry.h"
#include "LArUtil/LArProperties.h"
#include "LArUtil/SpaceChargeMicroBooNE.h"
#include "LArUtil/TimeService.h"
#include "SelectionTool/OpT0Finder/PhotonLibrary/PhotonVisibilityService.h"

// cluster
#include "cluster/DBSCAN.h"
#include "cluster/CoreFilter.h"
#include "cluster/CilantroPCA.h"

namespace larflow {

    QClusterCore::QClusterCore( const QCluster_t& qcluster ) :
    _cluster(&qcluster)
  {
    buildCore();
    fillClusterGapsUsingCorePCA();
  }

  void QClusterCore::buildCore() {

    _core.clear();
    
    int minneighbors = 3;
    int minclusterpoints = 3;
    float maxdist = 20.0;
    float fMaxDistFromPCAcore = 10.0;

    std::cout << "[larflow::QClusterCore::buildCore][DEBUG] Build core for QCLUSTER[" << _cluster->idx << "]" << std::endl;
    
    // const larutil::Geometry* geo = larutil::Geometry::GetME();
    // const larutil::LArProperties* larp = larutil::LArProperties::GetME();    
    // const float  driftv = larp->DriftVelocity();    
    std::cout << "[larflow::QClusterCore::buildCore][DEBUG] input cluster size=" << _cluster->size() << std::endl;
    
    std::vector< std::vector<float> > clusterpts;
    clusterpts.reserve(_cluster->size()); // need to define a non-copy way to do this ...
    for ( auto const& qhit : *_cluster ) {
      clusterpts.push_back( qhit.xyz );
    }
    std::cout << "[larflow::QClusterCore::buildCore][DEBUG] number of cluster points=" << clusterpts.size() << std::endl;
    
    // core filter
    CoreFilter corealgo( clusterpts, minneighbors, maxdist );
    std::cout << "[larflow::QClusterCore::buildCore][DEBUG] core filter split into =" << corealgo.getNumClusters() << std::endl;

    // we take a guess that the largest cluster is the core
    // SHOULD BE LARGEST AND STRAIGHTEST
    int largest_idx = corealgo.getIndexOfLargestCluster();
    std::cout << "[larflow::QClusterCore::buildCore][DEBUG] largest core idx=" << largest_idx << std::endl;
    if ( largest_idx<0 ) {
      // null core -- what to do
      assert(false);
      return;
    }
    
    std::vector<int> core_indices = corealgo.getClusterIndices( largest_idx );
    size_t largestcore_size = core_indices.size();
    std::cout << "[larflow::QClusterCore::buildCore][DEBUG] largest core size: " << largestcore_size << std::endl;
    std::vector< std::vector<float> > core(largestcore_size);

    
    for ( size_t icore=0; icore<core_indices.size(); icore++) {
      int idx = core_indices.at(icore);
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
    int joinsize = largestcore_size;
    int numsubclusts = (int)corealgo.getNumClusters()-1; // skip the last, that's the noise clusters
    for ( int iclust=0; iclust<numsubclusts; iclust++ ) {
      if ( iclust==largest_idx ) continue;
      std::cout << "test: " << iclust << std::endl;
      bool join2core = false;
      std::vector<int> subclustidx_v = corealgo.getClusterIndices(iclust);
      std::cout << "size=" << subclustidx_v.size() << " numclusters=" << corealgo.getNumClusters() << std::endl;      

      if ( subclustidx_v.size()<minclusterpoints )
	continue;

      for (auto& idx : subclustidx_v ) {
	std::cout << "[" << iclust << "] idx=" << idx << std::endl;
	Eigen::Map< Eigen::Vector3f > testpt( clusterpts[idx].data() );
	float dist = coreline.distance(testpt); // eigen is great
	if ( dist < fMaxDistFromPCAcore ) {
	  join2core = true;
	  break;
	}
      }
      
      if (join2core ) {
	joinlist.insert( iclust );
	joinsize += subclustidx_v.size();
      }
    }//end of loop over subclusters

    // collect final core points, get pca, sort by projection
    std::vector< std::vector<float> > finalcorepts;
    std::vector< int > finalcore_idx;
    finalcorepts.reserve(joinsize);
    finalcore_idx.reserve(joinsize);
    for (auto& idx : core_indices ) {
      finalcorepts.push_back( (*_cluster)[idx].xyz );
      finalcore_idx.push_back( idx );
    }
    for ( auto& iclust : joinlist )  {
      std::cout << " add non core iclustidx=" << iclust << std::endl;
      std::vector<int> noncoreidx = corealgo.getClusterIndices(iclust);
      if (noncoreidx.size()==0)
	continue;
      for (auto& idx : noncoreidx ) {
     	finalcorepts.push_back( (*_cluster)[idx].xyz );
	finalcore_idx.push_back( idx );
      }
    }

    
    // pca's
    CilantroPCA finalcorepca( finalcorepts );
    _pca_core = finalcorepca.getpcaxis();

    Eigen::Vector3f origin2( _pca_core.getAvePosition()[0],     _pca_core.getAvePosition()[1],     _pca_core.getAvePosition()[2] );
    Eigen::Vector3f vec2(    _pca_core.getEigenVectors()[0][0], _pca_core.getEigenVectors()[1][0], _pca_core.getEigenVectors()[2][0] );
    Eigen::ParametrizedLine< float, 3 > coreline2( origin2, vec2 );

    std::vector< ProjPoint_t > proj_v;
    for (int icore=0; icore<(int)finalcorepts.size(); icore++) {
      Eigen::Map< Eigen::Vector3f >  ept( finalcorepts[icore].data() );
      Eigen::Vector3f projpt = coreline2.projection( ept );
      float s = (projpt-origin2).norm();
      float coss = vec2.dot(projpt-origin2);
      ProjPoint_t pjpt;
      pjpt.idx = finalcore_idx[icore];;
      pjpt.s = ( coss<0 ) ? -s : s;
      proj_v.push_back( pjpt );
    }
    std::sort(proj_v.begin(), proj_v.end());

    // make core, make non-core points
    _core.reserve( joinsize );
    for (auto& projpt : proj_v ) {
      _core.push_back( (*_cluster)[projpt.idx] );
    }
    _core.idx = (*_cluster).idx;
    _core.mctrackid = (*_cluster).mctrackid;
    _core.truthmatched_flashidx = (*_cluster).truthmatched_flashidx;

    // not core
    for ( int iclust=0; iclust<(int)numsubclusts; iclust++ ) {
      auto it = joinlist.find(iclust);
      if ( it!=joinlist.end() ) continue;
      if ( corealgo.getClusterIndices(iclust).size()==0 ) continue;

      QCluster_t qnoncore;
      for (auto& idx : corealgo.getClusterIndices(iclust) )
    	qnoncore.push_back( (*_cluster)[idx] );
      qnoncore.idx = (*_cluster).idx;
      qnoncore.mctrackid = (*_cluster).mctrackid;
      qnoncore.truthmatched_flashidx = (*_cluster).truthmatched_flashidx;
      
      _noncore.emplace_back( std::move(qnoncore) );
    }
    
    // non-core pca
    _noncore_hits = 0;
    for ( int inoncore=0; inoncore<(int)_noncore.size(); inoncore++ ) {
      QCluster_t& qnoncore = _noncore[inoncore];
      
      std::vector< std::vector<float> > noncorepts(qnoncore.size());
      for ( int ihit=0; ihit<(int)qnoncore.size(); ihit++ ) {
	noncorepts[ihit] = qnoncore[ihit].xyz;
	_noncore_hits++;
      }
      CilantroPCA noncorepca( noncorepts );
      _pca_noncore.push_back( noncorepca.getpcaxis() );
    }//noncore loop

    std::cout << "[larflow::QClusterCore::buildCore][DEBUG] defined core" << std::endl;
  }//end of define core

  void QClusterCore::fillClusterGapsUsingCorePCA() {
    
    const float kGapMin_cm = 6.0; // cm
    const float kGapStepLenMin_cm = 1.0; // cm
    const float kGapStepLenMax_cm = 3.0; // cm

    larlite::pcaxis& pca = _pca_core;
    if ( pca.getEigenVectors().size()==0 ) {
      // not core
      return;
    }
    
    // we project points on the pca line
    // define eigen line
    Eigen::Vector3f origin( pca.getAvePosition()[0], pca.getAvePosition()[1], pca.getAvePosition()[2] );
    Eigen::Vector3f vec( pca.getEigenVectors()[0][0], pca.getEigenVectors()[1][0], pca.getEigenVectors()[2][0] );
    Eigen::ParametrizedLine< float, 3 > pcaline( origin, vec );
    //std::cout << "[LArFlowFlashMatch::fillClusterGapsUsingCorePCA][DEBUG] pca-origin=" << origin.transpose() << "  vec=" << vec.transpose() << std::endl;

    std::vector< ProjPoint_t > orderedline;
    orderedline.reserve( _core.size() );
    for ( size_t icore=0; icore<_core.size(); icore++ ) {
      Eigen::Map< Eigen::Vector3f >  ept( _core[icore].xyz.data() );
      Eigen::Vector3f projpt = pcaline.projection( ept );
      float s = (projpt-origin).norm();
      float coss = vec.dot(projpt-origin);
      ProjPoint_t pjpt;
      pjpt.idx = icore;
      pjpt.s = ( coss<0 ) ? -s : s;
      orderedline.push_back( pjpt );
    }
    std::sort( orderedline.begin(), orderedline.end() );

    // now loop through and look for gaps
    // keep track of average gap -- help us set the filler point density
    float avegap = 0.;
    int nfillgaps = 0;
    std::vector< int > gapstarts;
    for ( size_t ipt=0; ipt+1<orderedline.size(); ipt++ ) {

      // cannot have gap between two EXT pts
      
      float gap = fabs(orderedline[ipt+1].s - orderedline[ipt].s);
      //std::cout << " orderedline[" << ipt << "] gap=" << gap << " s0=" << orderedline[ipt].s << std::endl;
      if ( gap>kGapMin_cm )  {
	gapstarts.push_back( ipt );	
      }
      else {
	avegap += gap;
	nfillgaps++;
      }
    }
    if ( nfillgaps>0 )
      avegap /= float(nfillgaps);

    std::cout << "[QClusterCore::fillClusterGapsUsingCorePCA][INFO] number gaps to fill=" << gapstarts.size() << " avegap=" << avegap << std::endl;
    if ( nfillgaps==0 ) // no need to do anything!
      return;
    
    // add gap points
    _gapfill_qcluster.clear();
    for ( auto& idxgap : gapstarts ) {
      // get the projected point info
      ProjPoint_t& start = orderedline[idxgap];
      ProjPoint_t& end   = orderedline[idxgap+1];
      // get the points
      Eigen::Map< Eigen::Vector3f > startpt( _core[start.idx].xyz.data() );
      Eigen::Map< Eigen::Vector3f > endpt( _core[end.idx].xyz.data() );
      Eigen::ParametrizedLine< float, 3 > gapline = Eigen::ParametrizedLine<float,3>::Through( startpt, endpt );
      float gaplen = (endpt-startpt).norm();
      // step len
      float steplen = ( avegap < kGapStepLenMax_cm ) ? avegap : kGapStepLenMax_cm;
      steplen = ( steplen > kGapStepLenMin_cm ) ? steplen : kGapStepLenMin_cm;
      int nsteps = (int)(gaplen/steplen) + 1;
      steplen = gaplen/float(nsteps);
      float tickstart = _core[ start.idx ].tick;
      float tickend   = _core[ end.idx ].tick;
      float dticklen = (tickend-tickstart)/float(nsteps);
      for (int istep=0; istep<nsteps; istep++) {
	float s = steplen*((float)istep+0.5);
	Eigen::Vector3f gappt = gapline.pointAt( s );
	QPoint_t qpt;
	qpt.xyz.resize(3,0);
	for (int i=0; i<3; i++) qpt.xyz[i] = gappt(i);
	qpt.type = kGapFill;
	qpt.pixeladc = steplen;
	qpt.tick = tickstart + dticklen*((float)istep+0.5);
	_gapfill_qcluster.emplace_back( std::move(qpt) );
      }
    }//end of gap starts loop
    
  }//end of fillgaps

}
