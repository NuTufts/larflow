#ifndef __LARFLOW_NUVERTEX_FITTER_H__
#define __LARFLOW_NUVERTEX_FITTER_H__

/**
 * The purpose of this class is to optimize the position of 
 * the proposed vertex based on surrounding prongs.
 *
 * Inputs:
 * 1) vertex candidate locations (from NuVertexMaker)
 * 2) clusters associated to the cluster (clusters from various algorithms, but associated clusters from NuVertexMaker)
 *
 * Outputs:
 * 1) optimized vertex location
 * 2) prongs
 *
 * Optimization performed using a least squares fit
 * weighted by charge and larmatch confidence.
 * And a weak prior based on the original vertex location.
 *
 */

#include "larcv/core/Base/larcv_base.h"

#include "larcv/core/DataFormat/IOManager.h"
#include "DataFormat/storage_manager.h"

#include "NuVertexMaker.h"

namespace larflow {
namespace reco {

  class NuVertexFitter : public larcv::larcv_base {

  public:

    NuVertexFitter()
      : larcv::larcv_base("NuVertexFitter")
      {};

    virtual ~NuVertexFitter() {};


    void process( larcv::IOManager& iolcv, larlite::storage_manager& ioll,
                  larflow::reco::NuVertexMaker& vtxmaker );

  };
  

}
}

#endif
