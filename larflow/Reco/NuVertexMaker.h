#ifndef __NU_VERTEX_MAKER_H__
#define __NU_VERTEX_MAKER_H__


#include <string>
#include <map>

#include "TTree.h"
#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "larlite/DataFormat/storage_manager.h"
#include "larlite/DataFormat/pcaxis.h"
#include "larlite/DataFormat/larflow3dhit.h"
#include "larlite/DataFormat/larflowcluster.h"

#include "NuVertexCandidate.h"

namespace larflow {
namespace reco {

  /**
   * @ingroup Reco
   * @class NuVertexMaker
   * @brief Form Neutrino Vertex Candidates.
   *
   * Approach is to use Keypoints and rank candidates
   * based on number of track and shower segments that point back to it.
   *
   * Goal is to form seeds that will be used to build particle hypothesis graphs.
   *
   */  
  class NuVertexMaker : public larcv::larcv_base {

  public:

    NuVertexMaker();
    virtual ~NuVertexMaker() {};

    typedef enum { kRaw=0, kMerged, kVetoed, kFitted } OutStage_t;
    void setOutputStage( OutStage_t stage ) { _output_stage=stage; };
    
    void process( larcv::IOManager& ioman, larlite::storage_manager& ioll );

  protected:

    OutStage_t _output_stage;
    std::vector<NuVertexCandidate> _vertex_v; ///< initial vertex candidates
    std::vector<NuVertexCandidate> _merged_v; ///< after merging nearby vertices
    std::vector<NuVertexCandidate> _vetoed_v; ///< after filtering out candidates based on proximity to uncontained cosmic tracks
    std::vector<NuVertexCandidate> _fitted_v; ///< after prong fitting

  protected:

    std::map<std::string, larlite::event_larflow3dhit* > _keypoint_producers;      ///< map from tree name to event container for keypoints
    std::map<std::string, larlite::event_pcaxis* >       _keypoint_pca_producers;  ///< map from tree name to pca info for keypoints

    std::map<std::string, larlite::event_larflowcluster* > _cluster_producers;       ///< map from tree name to event container for larflowcluster
    std::map<std::string, larlite::event_pcaxis* >         _cluster_pca_producers;   ///< map from tree name to pca info for clusters
    std::map<std::string, larlite::event_track* >          _cluster_track_producers; ///< map from tree name to track line fit for track clusters
    std::map<std::string, NuVertexCandidate::ClusterType_t > _cluster_type;        ///< cluster type
    std::map<NuVertexCandidate::ClusterType_t, float>        _cluster_type_max_impact_radius; ///< max distance from cluster pca to vertex allowed, per class type
    std::map<NuVertexCandidate::ClusterType_t, float>        _cluster_type_max_gap; ///< maximum gap between vertex and start of cluster, per class type
    
  public:

    /** @brief add tree name to the list of vectors to get keypoint data */
    void add_keypoint_producer( std::string name ) {
      _keypoint_producers[name] = nullptr;
      _keypoint_pca_producers[name] = nullptr;
    };

    /** @brief add name and type to the list clusterat */    
    void add_cluster_producer( std::string name,
                               NuVertexCandidate::ClusterType_t ctype ) {
      _cluster_producers[name] = nullptr;
      _cluster_pca_producers[name] = nullptr;
      _cluster_track_producers[name] = nullptr;
      _cluster_type[name] = ctype;
    };

    /** @brief get initial candidates */
    const std::vector<NuVertexCandidate>& get_nu_candidates() const { return _vertex_v; };

    /** @brief get candidates after vetoing using cosmics */
    const std::vector<NuVertexCandidate>& get_vetoed_candidates() const { return _vetoed_v; };

    /** @brief get candidates after merging nearby candidates (in 3D space) */
    const std::vector<NuVertexCandidate>& get_merged_candidates() const { return _merged_v; };

    /** @brief get candidates after optimizing vertex position */
    const std::vector<NuVertexCandidate>& get_fitted_candidates() const { return _fitted_v; };            

    /** @brief get mutable initial candidates */
    std::vector<NuVertexCandidate>& get_mutable_nu_candidates() { return _vertex_v; };

    /** @brief get mutable candidates after vetoing using cosmics */
    std::vector<NuVertexCandidate>& get_mutable_vetoed_candidates() { return _vetoed_v; };

    /** @brief get mutable candidates after merging nearby candidates (in 3D space) */
    std::vector<NuVertexCandidate>& get_mutable_merged_candidates() { return _merged_v; };

    /** @brief get mutable candidates after optimizing vertex position */
    std::vector<NuVertexCandidate>& get_mutable_fitted_candidates() { return _fitted_v; };            

    /** @brief get mutable candidates specified by _output_stage flag */
    std::vector<NuVertexCandidate>& get_mutable_output_candidates();
    
    void clear();
    
  protected:

    void _createCandidates(larcv::IOManager& iolcv);
    void _set_defaults();
    void _score_vertex( NuVertexCandidate& vtx ); 
    void _merge_candidates();
    bool _attachClusterToCandidate( NuVertexCandidate& vertex,
                                    const larlite::larflowcluster& lfcluster,
                                    const larlite::pcaxis& lfpca,
                                    NuVertexCandidate::ClusterType_t ctype,
                                    std::string producer,
                                    int icluster,                                    
                                    bool apply_cut );
    void _cosmic_veto_candidates( larlite::storage_manager& ioll );
    void _refine_position( larcv::IOManager& iolcv,
                           larlite::storage_manager& ioll );
    
  protected:

    bool   _own_tree;    ///< own output tree, i.e. _ana_tree is not null.
    int    _ana_run;     ///< [ana tree variable] run number
    int    _ana_subrun;  ///< [ana tree variable] subrun number
    int    _ana_event;   ///< [ana tree variable] event number
    TTree* _ana_tree;    ///< ROOT tree to save output of algorithm
    bool   _apply_cosmic_veto;  ///< apply wirecell filter to reduce number of possible keypoints
    
  public:
    
    void make_ana_tree();
    void add_nuvertex_branch( TTree* tree );

    /** @brief Fill the _ana_tree with data from current event */
    void fill_ana_tree()  { if (_ana_tree) _ana_tree->Fill(); };

    /** @brief Write data saved in _ana_tree to file */
    void write_ana_tree() { if (_ana_tree) _ana_tree->Write(); };

    /** @brief set flag that if true, filters keypoint using proximity to boundary cosmic muon */
    void apply_cosmic_veto( bool applyveto ) { _apply_cosmic_veto=applyveto; };
  };
  
}
}

#endif
