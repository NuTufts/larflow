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

    struct Prong_t {
      std::vector< std::vector<float> > feat_v;
      const larlite::larflowcluster* orig_cluster;
      const larlite::pcaxis* orig_pcaxis;
      std::vector<float> endpt;
      std::vector<float> startpt;
    };

    void process( larcv::IOManager& iolcv, larlite::storage_manager& ioll,
                  const std::vector< larflow::reco::NuVertexCandidate >& vertex_v );
    const std::vector< std::vector<float> >& get_fitted_pos() { return _fitted_pos_v; };
    
  protected:

    void _fit_vertex( const std::vector<float>& initial_vertex_pos,
                      const std::vector<Prong_t>& prong_v,
                      std::vector<float>& fitted_pos,
                      float& delta_loss );


    std::vector< std::vector<float> > _fitted_pos_v;

  };
  

}
}

#endif
