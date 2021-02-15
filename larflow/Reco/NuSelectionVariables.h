#ifndef __LARFLOW_RECO_NU_SELECTION_VARIABLES_H__
#define __LARFLOW_RECO_NU_SELECTION_VARIABLES_H__

#include <vector>

namespace larflow {
namespace reco {

  /**
   * @brief A class that stores results from different algorithms
   *
   * Each instance corresponds to a NuVertexCandidate.
   *
   */
  class NuSelectionVariables
  {

  public:

    NuSelectionVariables()
      : max_proton_pid(0.0),
      dist2truevtx(1e6),
      truth_vtxFracNu(0.),
      isTruthMatchedNu(0)
      {};
    virtual ~NuSelectionVariables() {};


    /**
     * @brief struct to store track-level variables
     */
    struct TrackVar_t {
      
      float length;     ///< sum of track segment lengths

      // variables targeting proton ID
      float pca_ratio;  ///< ratio of 2nd to 1st PCA eigenvalue (measure of straightness)
      float proton_ll;  ///< likilihood ratio of proton to muon dq/dx curve (
      float frac_hip;   ///< fraction of spacepoints where majority of plane pixel is HIP class

      // results/summaries
      float protonid;   ///< reserved for some kind of proton score
      float muonid;     ///< reserved for muon id
      float pionid;     ///< reserved for pion id

      TrackVar_t()
      : length(0),
        pca_ratio(0),
        proton_ll(0),
        frac_hip(0),
        protonid(0),
        muonid(0),
        pionid(0)
      {};
      
    };
    std::vector< TrackVar_t > _track_var_v;

    // SUMMARY
    float max_proton_pid;

    // made by NuSelProngVars
    int   ntracks;
    int   nshowers;
    float max_shower_length;
    float max_track_length;
    int   max_shower_nhits;
    int   max_track_nhits;
    float min_shower_gap;
    float max_shower_gap;

    // made by NuSelVertexVars
    std::vector<int>   nabove_threshold_vertex_pix_v;
    std::vector<float> vertex_plane_charge_v;
    float vertex_hip_fraction;
    float vertex_charge_per_pixel;
    int   vertex_type;
    
    // TRUTH
    float dist2truevtx;    ///< distance to true vertex, sce applied to true vertex
    float truth_vtxFracNu; ///< fraction of all plane pixels on nu
    int isTruthMatchedNu;  ///< 1 if truth-matched. provides target for selection training.
    
  };
  
}
}

#endif
