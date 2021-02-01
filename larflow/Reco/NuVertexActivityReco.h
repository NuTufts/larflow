#ifndef __LARFLOW_RECO_NUVERTEX_ACTIVITYRECO_H__
#define __LARFLOW_RECO_NUVERTEX_ACTIVITYRECO_H__

#include "DataFormat/storage_manager.h"
#include "DataFormat/larflow3dhit.h"
#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/EventImage2D.h"
#include "ublarcvapp/MCTools/LArbysMC.h"
#include "cluster_functions.h"

namespace larflow {
namespace reco {

  /**
   * @ingroup Reco
   * @class NuVertexActivityReco
   *
   * Uses larmatch points to find 3D-consistent point-like deposition.
   * Filter out background and real candiates using larmatch keypoint candidates.
   * Emphasis is end of showers.
   *
   */
  class NuVertexActivityReco : public larcv::larcv_base {
    
  public:
    NuVertexActivityReco()
      : larcv::larcv_base("NuVertexActivityReco"),
      _va_ana_tree(nullptr),
      _kown_tree(false)
        {};
    virtual ~NuVertexActivityReco() { if ( _kown_tree && _va_ana_tree ) delete _va_ana_tree; };

    void process( larcv::IOManager& iolcv, larlite::storage_manager& ioll );
    void make_tree();
    void bind_to_tree(TTree* tree );
    void write_tree() { _va_ana_tree->Write(); };
    void clear_ana_variables();
    void fill_tree() { _va_ana_tree->Fill(); };
    void calcTruthVariables( larlite::storage_manager& ioll,
                             const ublarcvapp::mctools::LArbysMC& truedata );

  protected:
    
    void makeClusters( larlite::storage_manager& ioll,
                       std::vector<larflow::reco::cluster_t>& cluster_v,
                       const float larmatch_threshold );

    std::vector<larlite::larflow3dhit>
      findVertexActivityCandidates( larlite::storage_manager& ioll,
                                    larcv::IOManager& iolcv,
                                    std::vector<larflow::reco::cluster_t>& cluster_v,
                                    const float va_threshold,
                                    std::vector< std::vector<float> >& vtxact_dir_v );
      
    
    std::vector<float> calcPlanePixSum( const larlite::larflow3dhit& hit,
                                        const std::vector<larcv::Image2D>& adc_v );


    void analyzeVertexActivityCandidates( larlite::larflow3dhit& va_cand,
                                          std::vector<float>& va_dir,
                                          std::vector<larflow::reco::cluster_t>& cluster_v,
                                          larlite::storage_manager& ioll,
                                          larcv::IOManager& iolcv,
                                          const float min_dist2cluster );
    
    TTree* _va_ana_tree;  //< event level tree with data for each reco VA candidate
    bool _kown_tree;
    std::vector< std::vector<float> > pca_dir_vv;
    std::vector<int> nbackwards_shower_pts;
    std::vector<int> nbackwards_track_pts;
    std::vector<int> nforwards_shower_pts;
    std::vector<int> nforwards_track_pts;
    std::vector<float> dist_closest_forwardshower;
    std::vector<float> shower_likelihood;
    std::vector<float> dist2truescevtx;
    float min_dist2truescevtx;
    
    
  };

  
}
}

#endif
