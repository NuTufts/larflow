#include "ShowerRecoKeypoint.h"

#include "larcv/core/DataFormat/EventImage2D.h"
#include "PCACluster.h"
#include "nlohmann/json.hpp"

#include "DataFormat/larflow3dhit.h"
#include "DataFormat/larflowcluster.h"
#include "DataFormat/pcaxis.h"

#include <ctime>
#include <fstream>

namespace larflow {
namespace reco {

  void ShowerRecoKeypoint::process( larcv::IOManager& iolc, larlite::storage_manager& ioll ) {

    // we process shower clusters produced by PCA cluster algo
    // steps:
    // (1) find trunk candidates:
    //      - we contour the shower pixels and look for straight segments
    //      - we gather track clusters that are connected to the shower clusters
    // (2) we build a shower hypothesis from the trunks:
    //      - we add points along the pca-axis of the cluster
    //      - does one end of the trunk correspond to the end of the assembly? define as start point
    //      - shower envelope expands from start
    //      - trunk pca and assembly pca are aligned
    // (3) choose the best shower hypothesis that has been formed
    
    // get shower larflow hits (use SplitHitsBySSNet)
    larlite::event_larflow3dhit* shower_lfhit_v
      = (larlite::event_larflow3dhit*)ioll.get_data( larlite::data::kLArFlow3DHit, _ssnet_lfhit_tree_name );

    // make shower clusters
    float maxdist = 2.0;
    int minsize = 20;
    int maxkd = 10;
    std::vector<cluster_t> cluster_v;
    cluster_larflow3dhits( *shower_lfhit_v, cluster_v, maxdist, minsize, maxkd );
    

    std::clock_t begin_process = std::clock();
    LARCV_INFO() << "start" << std::endl;
    LARCV_INFO() << "num larflow hits from [" << _ssnet_lfhit_tree_name << "]: " << shower_lfhit_v->size() << std::endl;
    LARCV_INFO() << "num shower clusters:  " << cluster_v.size() << std::endl;    

    // now for each shower cluster, we find some trunk candidates.
    // can have any number of such candidates per shower cluster
    // we only analyze clusters with a first pc-axis length > 1.0 cm
    std::vector< const cluster_t* > trunk_candidates_v;

    int idx = -1;
    for ( auto& showercluster : cluster_v ) {
      idx++;

      cluster_pca( showercluster );
      
      // metrics to choose shower trunk candidates
      // length
      float len =  showercluster.pca_len;

      // pca eigenvalue [1]/[0] ratio -- to ensure straightness
      float eigenval_ratio = showercluster.pca_eigenvalues[1]/showercluster.pca_eigenvalues[0];
      
      LARCV_DEBUG() << "shower cluster[" << idx << "]"
                    << " pca-len=" << len << " cm,"
                    << " pca-eigenval-ratio=" << eigenval_ratio
                    << std::endl;
      
      if ( len<1.0 ) continue;
      //if ( eigenval_ratio<0.1 ) continue;

      trunk_candidates_v.push_back( &showercluster );
    }

    LARCV_INFO() << "num of trunk candidates: " << trunk_candidates_v.size() << std::endl;

    std::clock_t end_process = std::clock();
    LARCV_INFO() << "[ShowerRecoKeypoint::process] end; elapsed = "
                 << float(end_process-begin_process)/CLOCKS_PER_SEC << " secs"      
                 << std::endl;

    larlite::event_larflowcluster* evout_shower_cluster_v
      = (larlite::event_larflowcluster*)ioll.get_data( larlite::data::kLArFlowCluster, "showerkp" );
    for ( auto const& pc : trunk_candidates_v ) {
      larlite::larflowcluster lfcluster;
      for (auto const& idx : pc->hitidx_v ) {
        lfcluster.push_back( shower_lfhit_v->at(idx) );
      }
      evout_shower_cluster_v->emplace_back( std::move(lfcluster) );
    }

    larlite::event_pcaxis* evout_shower_pca_v
      = (larlite::event_pcaxis*)ioll.get_data( larlite::data::kPCAxis, "showerkp" );
    int pcidx = 0;
    for ( auto const& pc : trunk_candidates_v ) {

      larlite::pcaxis pca = cluster_make_pcaxis( *pc, pcidx );

      evout_shower_pca_v->emplace_back( std::move(pca) );
      pcidx++;
    }

  }

  /**
   * 
   * match keypoints to shower clusters, use to define the trunk
   *
   * we match keypoints to shower clusters
   * for each keypoint assigned to cluster, define 1,3,5 cm hit cluster around each keypoint
   * going from 5,3,1 cm clusters, accept pca-axis based on eigenvalue ratio
   * 
   * use log likelihood function to pick best key-point trunk
   * output is shower cluster, keypoint, and trunk cluster
   * 
   */
  void ShowerRecoKeypoint::_reconstructClusterTrunks( const std::vector<const cluster_t*>&    showercluster_v,
                                                      const std::vector<const larlite::larflow3dhit*>& keypoint_v )
  {

    
  }
  
}
}
