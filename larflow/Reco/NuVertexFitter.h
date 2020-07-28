#ifndef __LARFLOW_NUVERTEX_FITTER_H__
#define __LARFLOW_NUVERTEX_FITTER_H__


#include "larcv/core/Base/larcv_base.h"

#include "larcv/core/DataFormat/IOManager.h"
#include "DataFormat/storage_manager.h"

#include "NuVertexMaker.h"

namespace larflow {
namespace reco {

  /**
   * @ingroup Reco
   * @class NuVertexFitter
   * @brief optimize the position of the proposed vertex based on surrounding prongs
   *
   *
   * Inputs:
   * @verbatim embed:rst:leading-asterisk
   *  * vertex candidate locations (from NuVertexMaker)
   *  * clusters associated to the cluster (clusters from various algorithms, but associated clusters from NuVertexMaker)
   * @endverbatim
   *
   * Outputs:
   * @verbatim embed:rst:leading-asterisk
   *  * optimized vertex location
   *  * prongs
   * @endverbatim
   *
   * Optimization performed using a least squares fit
   * weighted by charge and larmatch confidence.
   * And a weak prior based on the original vertex location.
   *
   */  
  class NuVertexFitter : public larcv::larcv_base {

  public:

    NuVertexFitter()
      : larcv::larcv_base("NuVertexFitter")
      {};

    virtual ~NuVertexFitter() {};

    /**
     * @struct Prong_t
     * @brief Internal struct representing prongs coming from vertex
     */
    struct Prong_t {
      std::vector< std::vector<float> > feat_v;    ///< stores vectors with (x,y,z,lm,q)
      const larlite::larflowcluster* orig_cluster; ///< pointer to larflowcluster feat_v derives from
      const larlite::pcaxis* orig_pcaxis;          ///< pointer to principle component analysis of orig_cluster
      std::vector<float> endpt;                    ///< end position
      std::vector<float> startpt;                  ///< start position
    };

    void process( larcv::IOManager& iolcv, larlite::storage_manager& ioll,
                  const std::vector< larflow::reco::NuVertexCandidate >& vertex_v );

    /** @brief get fitted positions for all vertices provided */
    const std::vector< std::vector<float> >& get_fitted_pos() { return _fitted_pos_v; };
    
  protected:

    void _fit_vertex( const std::vector<float>& initial_vertex_pos,
                      const std::vector<Prong_t>& prong_v,
                      std::vector<float>& fitted_pos,
                      float& delta_loss );


    std::vector< std::vector<float> > _fitted_pos_v; ///< container of fitted positions for each vertex

  };
  

}
}

#endif
