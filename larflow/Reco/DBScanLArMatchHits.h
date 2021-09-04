#ifndef __LARFLOW_DBSCAN_LARMATCH_H__
#define __LARFLOW_DBSCAN_LARMATCH_H__

#include <vector>

#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/Image2D.h"
#include "larlite/DataFormat/storage_manager.h"
#include "larlite/DataFormat/larflowcluster.h"

#include "cluster_functions.h"

namespace larflow {
namespace reco {

  /**
   * @ingroup Reco
   * @class DBScanLArMatchHits
   * @brief cluster larflow3d hits using dbscan
   *
   * Basically an interface between IO managers
   * and the cluster_sdbscan_larflow3dhits() routine,
   * which uses Simple DBScan from ublarcvapp.
   *
   */
  class DBScanLArMatchHits : public larcv::larcv_base {
  public:

    /** @brief default constructor */
    DBScanLArMatchHits()
      : _input_larflowhit_tree_name("larmatch"),
      _out_cluster_tree_name("dblarmatch"),
      _min_larmatch_score(0.0),
      _maxdist(10.0),
      _minsize(5),
      _maxkd(10)
      {};
    virtual ~DBScanLArMatchHits() {};

    void process( larcv::IOManager& iolc, larlite::storage_manager& ioll );

    /** @brief set minimum larmatch score to include hits */
    void set_min_larmatch_score( float min_score ) { _min_larmatch_score = min_score; };

    /** @brief set parameters that control dbscan behavior */
    void set_dbscan_pars( float maxdist, int minsize, int maxkd ) {
      _maxdist = maxdist;
      _minsize = minsize;
      _maxkd = maxkd;
    };

  protected:    
    

    larlite::larflowcluster makeLArFlowCluster( cluster_t& cluster,
                                                const std::vector<larlite::larflow3dhit>& source_lfhit_v );

    cluster_t absorb_nearby_hits( const cluster_t& cluster,
                                  const std::vector<larlite::larflow3dhit>& hit_v,
                                  std::vector<int>& used_hits_v,
                                  float max_dist2line );
    
    void makeCluster( const std::vector<larlite::larflow3dhit>& inputhits,
                      std::vector<cluster_t>& output_cluster_v,
                      std::vector<int>& used_hits_v );

  protected:
    
    // PARAMETERS

    std::string _input_larflowhit_tree_name; ///< name of tree to get larflow3dhit data from
    std::string _out_cluster_tree_name;      ///< name of tree which will store larlite::larflowcluster objects
    float _min_larmatch_score;               ///< larmatch score threshold for including points in clustering
    float _maxdist;                          ///< max distance value used in dbscan 
    int   _minsize;                          ///< minimum size of cluster
    int   _maxkd;                            ///< max number of neighbors a spacepoint can have in dbscan 

  public:

    /** @brief set the input tree name for larflow3dhit container */
    void set_input_larflowhit_tree_name( std::string treename )
    { _input_larflowhit_tree_name=treename; };

    /** @brief set the output tree name where larflowcluster objects will be stored */
    void set_output_larflowhit_tree_name( std::string treename )
    { _out_cluster_tree_name = treename; };
    
  };

}
}

#endif
