#ifndef __LARFLOW_RECO_NUVERTEX_CANDIDATE_H__
#define __LARFLOW_RECO_NUVERTEX_CANDIDATE_H__


#include <string>
#include <vector>
#include "DataFormat/track.h"
#include "DataFormat/larflowcluster.h"
#include "DataFormat/pcaxis.h"
#include "TLorentzVector.h"

namespace larflow {
namespace reco {

  /**
   * @ingroup Reco
   * @class NuVertexCandidate
   * @brief class to store neutrino interaction vertices and their constituents
   *
   * Made by NuVertexMaker.
   *
   */
  class NuVertexCandidate {

  public:

    NuVertexCandidate() {};
    virtual ~NuVertexCandidate() {};

    /**
     * @brief Type of Vertex Candidate
     */
    typedef enum { kTrack=0, kShowerKP, kShower } ClusterType_t;

    /**
     * @struct VtxCluster_t
     * @brief structure representing particle cluster associated to vertex
    */    
    struct VtxCluster_t {
      std::string producer;    ///< larflowcluster tree name this cluster came from
      int index;               ///< the cluster's index in the cluster container
      std::vector<float> dir;  ///< direction along first principle component
      std::vector<float> pos;  ///< start position
      float gap;               ///< distance from vertex
      float impact;            ///< distance of first pc axis to the vertex position
      int npts;                ///< number of points in cluster
      ClusterType_t type;      ///< type of cluster
    };
    
    std::string keypoint_producer;  ///< name of tree containing keypoints used to seed candidates
    int keypoint_index;             ///< index of vertex candidate in container above
    int keypoint_type;              ///< keypoint type
    std::vector<float> pos;         ///< keypoint position
    int row;                        ///< vertex row
    int tick;                       ///< vertex tick
    std::vector<int> col_v;         ///< image columns
    std::vector< VtxCluster_t >    cluster_v; ///< clusters assigned to vertex
    std::vector< larlite::pcaxis > cluster_pca_v; ///< cluster pca assigned to vertex
    float score;                    ///< vertex candidate score based on number of clusters assigned and the impact parameter of each cluster
    float maxScore;                 ///< max cluster impact parameter score for all assigned clusters
    float avgScore;                 ///< average cluster impact parameter score for all assigned clusters
    float netScore;                 ///< keypoint score (max score if vertex formed from merged candidates)
    float netNuScore;               ///< keypoint neutrino score (max score, considering only neutrino keypoint types, if formed from merged candidates)

    // TRACK PRONGS AND VARIABLES
    std::vector<larlite::track>  track_v;     ///< track candidates
    std::vector<larlite::larflowcluster>  track_hitcluster_v;  ///< track candidates
    std::vector<float>           track_len_v;       ///< length of track
    std::vector< std::vector<float> > track_dir_v;  ///< direction of track, using points near vertex
    std::vector<float>           track_kemu_v;      ///< range-based ke assuming muon (from NuTrackKinematics)
    std::vector<float>           track_keproton_v;  ///< range-based ke assuming proton (from NuTrackKinematics)
    std::vector<TLorentzVector>  track_pmu_v;       ///< range-based momentum assuming muon (from NuTrackKinematics)
    std::vector<TLorentzVector>  track_pproton_v;   ///< range-based momentum assuming proton (from NuTrackKinematics)
    std::vector<float>           track_muid_v;      ///< muon likelihood   (from TrackdQdx)
    std::vector<float>           track_protonid_v;  ///< proton likelihood (from TrackdQdx)
    std::vector<float>           track_mu_vs_proton_llratio_v; // muon/proton likelihood ratio
    
    std::vector<larlite::larflowcluster> shower_v; ///< shower candidates
    std::vector<larlite::track>  shower_trunk_v;   ///< line for shower trunk for plotting
    std::vector<larlite::pcaxis> shower_pcaxis_v;  ///< pc-axis of whole shower cluster
    std::vector< std::vector<float> > shower_plane_pixsum_vv; ///< pixel sum of showers, a value for each plane
    std::vector< std::vector<TLorentzVector> > shower_plane_mom_vv; ///< energy of showers, a value for each plane
    std::vector< std::vector<float> > shower_plane_dqdx_vv;  ///< dqdx of shower trunk, a value for each plane

    /** @brief comparator to sort candidates by highest score */
    bool operator<(const NuVertexCandidate& rhs) const {
      if ( score>rhs.score ) return true;
      return false;
    };
    
  };
  

  

}
}

#endif
