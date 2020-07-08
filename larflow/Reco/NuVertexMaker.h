#ifndef __NU_VERTEX_MAKER_H__
#define __NU_VERTEX_MAKER_H__

/**
 * Form Neutrino Vertex Candidates.
 * Approach is to use Keypoints and rank candidates
 * based on number of track and shower segments that point back to it.
 *
 * Goal is to form seed to build particle hypothesis graph.
 *
 */

#include <string>
#include <map>

#include "TTree.h"
#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "DataFormat/storage_manager.h"
#include "DataFormat/pcaxis.h"
#include "DataFormat/larflow3dhit.h"
#include "DataFormat/larflowcluster.h"

#include "NuVertexCandidate.h"

namespace larflow {
namespace reco {

  class NuVertexMaker : public larcv::larcv_base {

  public:

    NuVertexMaker();
    virtual ~NuVertexMaker() {};

    void process( larcv::IOManager& ioman, larlite::storage_manager& ioll );

  protected:

    std::vector<NuVertexCandidate> _vertex_v; ///< initial vertex candidates
    std::vector<NuVertexCandidate> _merged_v; ///< after merging nearby vertices
    std::vector<NuVertexCandidate> _vetoed_v; ///< after filtering out candidates based on proximity to uncontained cosmic tracks

  protected:

    std::map<std::string, larlite::event_larflow3dhit* > _keypoint_producers;
    std::map<std::string, larlite::event_pcaxis* >       _keypoint_pca_producers;

    std::map<std::string, larlite::event_larflowcluster* > _cluster_producers;
    std::map<std::string, larlite::event_pcaxis* >         _cluster_pca_producers;
    std::map<std::string, NuVertexCandidate::ClusterType_t > _cluster_type;
    std::map<NuVertexCandidate::ClusterType_t, float>        _cluster_type_max_impact_radius;
    std::map<NuVertexCandidate::ClusterType_t, float>        _cluster_type_max_gap;
    
  public:

    void add_keypoint_producer( std::string name ) {
      _keypoint_producers[name] = nullptr;
      _keypoint_pca_producers[name] = nullptr;
    };

    void add_cluster_producer( std::string name,
                               NuVertexCandidate::ClusterType_t ctype ) {
      _cluster_producers[name] = nullptr;
      _cluster_pca_producers[name] = nullptr;
      _cluster_type[name] = ctype;
    };

    const std::vector<NuVertexCandidate>& get_nu_candidates() const { return _vertex_v; };
    
    void clear();
    
  protected:

    void _createCandidates();
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
    
  protected:

    bool   _own_tree;
    int    _ana_run;
    int    _ana_subrun;
    int    _ana_event;
    TTree* _ana_tree;
    bool   _apply_cosmic_veto;
    
  public:
    
    void make_ana_tree();
    void add_nuvertex_branch( TTree* tree );
    void fill_ana_tree()  { if (_ana_tree) _ana_tree->Fill(); };
    void write_ana_tree() { if (_ana_tree) _ana_tree->Write(); };
    void apply_cosmic_veto( bool applyveto ) { _apply_cosmic_veto=applyveto; };
  };
  
}
}

#endif
