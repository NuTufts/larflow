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
      ntracks(0),
      nshowers(0),
      max_shower_length(0),
      max_track_length(0),
      max_shower_nhits(0),
      max_track_nhits(0),
      min_shower_gap(0),
      max_shower_gap(0),
      closest_shower_ll(0),
      largest_shower_ll(0),
      closest_shower_avedqdx(0),
      largest_shower_avedqdx(0),
      frac_trackhits_on_cosmic(0),
      frac_showerhits_on_cosmic(0),
      frac_allhits_on_cosmic(0),
      nshower_pts_on_cosmic(0),
      ntrack_pts_on_cosmic(0),
      nplanes_connected(0),
      dist2truevtx(1e6),
      truth_vtxFracNu(0),
      isTruthMatchedNu(0)
      {
        plane_connected_on_pass.resize(3,-1);
      };
    virtual ~NuSelectionVariables() {
    };


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

    /** @brief struct to store shower-level variables */
    struct ShowerVar_t {
      float dqdx_ave; ///< filled by NuSelShowerTrunkAna
      float llshower; ///< filled by NuSelShowerTrunkAna
    };
    std::vector< ShowerVar_t > _shower_var_v;

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
    float min_track_gap;
    float max_track_gap;

    // filled by NuSelShowerTrunkAna
    float closest_shower_ll; 
    float largest_shower_ll; 
    float closest_shower_avedqdx; 
    float largest_shower_avedqdx;

    // filled by NuSelShowerGapAna2D;
    std::vector<int> plane_connected_on_pass;
    int nplanes_connected;

    // made by NuSelVertexVars
    std::vector<int>   nabove_threshold_vertex_pix_v;
    std::vector<float> vertex_plane_charge_v;
    float vertex_hip_fraction;
    float vertex_charge_per_pixel;
    int   vertex_type;

    // filled by NuSelWCTaggerOverlap
    float frac_trackhits_on_cosmic;
    float frac_showerhits_on_cosmic;
    float frac_allhits_on_cosmic;
    int nshower_pts_on_cosmic;
    int ntrack_pts_on_cosmic;

    // filled by NuSelCosmicTagger
    // track-track pair variables
    /* float showercosmictag_mindwall_dwall; */
    /* float showercosmictag_mindwall_costrack; */
    /* float showercosmictag_maxbacktoback_dwall; */
    /* float showercosmictag_maxbacktoback_costrack; */
    /* // track-shower pair variables */
    /* float showercosmictag_maxboundarytrack_length; */
    /* float showercosmictag_maxboundarytrack_verticalcos; */
    /* float showercosmictag_maxboundarytrack_showercos; */

    // filled by TrackForwardBackwardLL
    //float backmu_forwproton_llr;

    // filled by NuSelUnrecoCharge
    std::vector<int>   intime_count_v; /// number of in-time pixels per plane
    std::vector<int>   unreco_count_v; /// number of un-reconstructed pixels per plane
    std::vector<float> unreco_fraction_v; /// fraction of in-time pixels, un-reco'd
    
    // TRUTH
    float dist2truevtx;    ///< distance to true vertex, sce applied to true vertex
    float truth_vtxFracNu; ///< fraction of all plane pixels on nu
    int isTruthMatchedNu;  ///< 1 if truth-matched. provides target for selection training.

    
    
  };
  
}
}

#endif
