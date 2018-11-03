#ifndef __QCLUSTER_COMPOSITE_H__
#define __QCLUSTER_COMPOSITE_H__

// ROOT
#include "TGraph.h"

// eigen
#include <Eigen/Dense>

// flashmatch
#include "FlashMatchTypes.h"
#include "QClusterCore.h"

namespace larflow {

  class QClusterComposite {

  public:

    QClusterComposite( const QCluster_t& source_cluster );
    virtual ~QClusterComposite() {};

    const QCluster_t*   _cluster;  // source cluster data
    QClusterCore        _core;     // core cluster found using PCA analysis of source

    QCluster_t _entering_qcluster; // extension of cluster along "entering" end and core 1st pc
    QCluster_t _exiting_qcluster;  // extension of cluster along "exiting" end and core 1st pc
    Eigen::Vector3f _posfront; // top end of core
    Eigen::Vector3f _posback; // bottom end of core
    Eigen::Vector3f _pcavec;
    Eigen::Vector3f _centerpos;

    float getMinTick() const { return 0.; };
    float getMaxTick() const { return 0.; };
    
    // generate a hypothesis.
    // the data flash tells us which part of the extension to use
    FlashCompositeHypo_t generateFlashCompositeHypo( const FlashData_t& flash, bool apply_ext ) const;
    std::vector< TGraph > getTGraphs( float xoffset ) const;
  protected:

    // extend the core to the detector edge from the most likely entering end
    //  and past the TPC boundaries in so far as it improves the flash-match
    float _fExtStepLen;
    void ExtendEnteringEnd();
    void ExtendExitingEnd();
    


  };

}

#endif
