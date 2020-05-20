#include "DBScanLArMatchHits.h"

#include "cluster_functions.h"

#include "nlohmann/json.hpp"
#include "ublarcvapp/ContourTools/ContourClusterAlgo.h"
#include "ublarcvapp/dbscan/DBScan.h"
#include "larcv/core/DataFormat/EventImage2D.h"

#include "DataFormat/larflow3dhit.h"
#include "DataFormat/larflowcluster.h"

#include "TRandom3.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <ctime>

namespace larflow {
namespace reco {

  void DBScanLArMatchHits::process( larcv::IOManager& iolcv, larlite::storage_manager& ioll ) {

    larlite::event_larflow3dhit* ev_lfhits
      = (larlite::event_larflow3dhit*)ioll.get_data( larlite::data::kLArFlow3DHit, _input_larflowhit_tree_name );

    // cluster track hits
    std::vector<int> used_hits_v;
    std::vector<cluster_t> cluster_v;
    makeCluster( *ev_lfhits, cluster_v, used_hits_v );
    
    // form clusters of larflow hits for saving
    larlite::event_larflowcluster* evout_lfcluster 
      = (larlite::event_larflowcluster*)ioll.get_data( larlite::data::kLArFlowCluster, _out_cluster_tree_name );
    larlite::event_pcaxis*         evout_pcaxis
      = (larlite::event_pcaxis*)        ioll.get_data( larlite::data::kPCAxis,         _out_cluster_tree_name );
    
    int cidx = 0;
    for ( auto& c : cluster_v ) {
      // cluster of hits
      larlite::larflowcluster lfcluster = makeLArFlowCluster( c, *ev_lfhits );
      evout_lfcluster->emplace_back( std::move(lfcluster) );
      // pca-axis
      larlite::pcaxis llpca = cluster_make_pcaxis( c, cidx );
      evout_pcaxis->push_back( llpca );
      cidx++;
    }

    // make noise cluster
    larlite::event_larflowcluster* evout_noise_lfcluster
      = (larlite::event_larflowcluster*)ioll.get_data( larlite::data::kLArFlowCluster, _out_cluster_tree_name+"noise" );    
    larlite::larflowcluster lfnoise;
    cluster_t noise_cluster;
    for ( size_t i=0; i<ev_lfhits->size(); i++ ) {
      auto& hit = (*ev_lfhits)[i];
      if ( used_hits_v[i]==0 ) {
        std::vector<float> pt = { hit[0], hit[1], hit[2] };
        std::vector<int> coord_v = { hit.targetwire[0], hit.targetwire[1], hit.targetwire[2], hit.tick };
        noise_cluster.points_v.push_back( pt  );
        noise_cluster.imgcoord_v.push_back( coord_v );
        noise_cluster.hitidx_v.push_back( i );
        lfnoise.push_back( hit );
      }
    }
    cluster_pca( noise_cluster );
    larlite::pcaxis noise_pca = cluster_make_pcaxis( noise_cluster );
    evout_noise_lfcluster->push_back( lfnoise );
    
  }

  larlite::larflowcluster DBScanLArMatchHits::makeLArFlowCluster( cluster_t& cluster,
                                                                  const std::vector<larlite::larflow3dhit>& source_lfhit_v ) {
    
    larlite::larflowcluster lfcluster;
    lfcluster.reserve( cluster.points_v.size() );

    for ( size_t ii=0; ii<cluster.ordered_idx_v.size(); ii++ ) {

      int ihit = cluster.ordered_idx_v[ii];
        
      larlite::larflow3dhit lfhit;
      lfhit.resize(3,0); // (x,y,z,pixU,pixV,pixY,matchprob)

      // transfer 3D points
      for (int i=0; i<3; i++) lfhit[i] = cluster.points_v[ihit][i];

      // transfer image coordates
      lfhit.srcwire = cluster.imgcoord_v[ihit][2];
      lfhit.targetwire.resize(3,0);
      lfhit.targetwire[0] = cluster.imgcoord_v[ihit][0];
      lfhit.targetwire[1] = cluster.imgcoord_v[ihit][1];
      lfhit.targetwire[2] = cluster.imgcoord_v[ihit][2];
      lfhit.tick          = cluster.imgcoord_v[ihit][3];
      
      lfcluster.emplace_back( std::move(lfhit) );
    }//end of hit loop
    
    return lfcluster;
  }

  cluster_t DBScanLArMatchHits::absorb_nearby_hits( const cluster_t& cluster,
                                                    const std::vector<larlite::larflow3dhit>& hit_v,
                                                    std::vector<int>& used_hits_v,
                                                    float max_dist2line ) {

    cluster_t newcluster;
    int nused = 0;
    for ( size_t ihit=0; ihit<hit_v.size(); ihit++ ) {

      auto const& hit = hit_v[ihit];

      if ( used_hits_v[ ihit ]==1 ) continue;
      
      // apply quick bounding box test
      if ( hit[0] < cluster.bbox_v[0][0] || hit[0]>cluster.bbox_v[0][1]
           || hit[1] < cluster.bbox_v[1][0] || hit[1]>cluster.bbox_v[1][1]
           || hit[2] < cluster.bbox_v[2][0] || hit[2]>cluster.bbox_v[2][1] ) {
        continue;
      }

      // else calculate distance from pca-line
      float dist2line = cluster_dist_from_pcaline( cluster, hit );

      if ( dist2line < max_dist2line ) {
        std::vector<float> pt = { hit[0], hit[1], hit[2] };
        std::vector<int> coord_v = { hit.targetwire[0], hit.targetwire[1], hit.targetwire[2], hit.tick };
        newcluster.points_v.push_back( pt );
        newcluster.imgcoord_v.push_back( coord_v );
        newcluster.hitidx_v.push_back( ihit );
        used_hits_v[ihit] = 1;
        nused++;
      }
      
    }

    if (nused>=10 ) {
      std::cout << "[absorb_nearby_hits] cluster absorbed " << nused << " hits" << std::endl;      
      cluster_pca( newcluster );
    }
    else {
      // throw them back
      std::cout << "[absorb_nearby_hits] cluster hits " << nused << " below threshold" << std::endl;            
      for ( auto& idx : newcluster.hitidx_v )
        used_hits_v[idx] = 0;
      newcluster.points_v.clear();
      newcluster.imgcoord_v.clear();
      newcluster.hitidx_v.clear();
    }

    
    return newcluster;
  }

  void DBScanLArMatchHits::makeCluster( const std::vector<larlite::larflow3dhit>& inputhits,
                                        std::vector<cluster_t>& output_cluster_v,
                                        std::vector<int>& used_hits_v ) {

    const int max_pts_to_cluster = 30000;
    
    TRandom3 rand(12345);

    used_hits_v.resize( inputhits.size(), 0 );
    output_cluster_v.clear();
    
    // count points remaining
    int total_pts_remaining = 0;
    // on first pass, all points remain
    total_pts_remaining = (int)inputhits.size();

    // downsample points, if needed
    std::vector<larlite::larflow3dhit> downsample_hit_v;
    downsample_hit_v.reserve( max_pts_to_cluster );

    float downsample_fraction = (float)max_pts_to_cluster/(float)total_pts_remaining;
    if ( total_pts_remaining>max_pts_to_cluster ) {
      for ( size_t ihit=0; ihit<inputhits.size(); ihit++ ) {
        if ( used_hits_v[ihit]==0 ) {
          if ( rand.Uniform()<downsample_fraction ) {
            downsample_hit_v.push_back( inputhits[ihit] );
          }
        }
      }
    }
    else {
      for ( size_t ihit=0; ihit<inputhits.size(); ihit++ ) {
        if ( used_hits_v[ihit]==0 ) {
          downsample_hit_v.push_back( inputhits[ihit] );
        }
      }
    }

    LARCV_INFO() << inputhits.size() << " hits downsampled to " << downsample_hit_v.size() << std::endl;

    // cluster these hits
    std::vector<larflow::reco::cluster_t> cluster_pass_v;
    larflow::reco::cluster_sdbscan_larflow3dhits( downsample_hit_v, cluster_pass_v, _maxdist, _minsize, _maxkd ); // external implementation, seems best
    larflow::reco::cluster_runpca( cluster_pass_v );

    // we then absorb the hits around these clusters
    for ( auto const& ds_cluster : cluster_pass_v ) {
      cluster_t dense_cluster = absorb_nearby_hits( ds_cluster,
                                                    inputhits,
                                                    used_hits_v,
                                                    10.0 );
      if ( dense_cluster.points_v.size()>0 ) 
        output_cluster_v.emplace_back( std::move(dense_cluster) );
    }
    int nused_tot = 0;
    for ( auto& used : used_hits_v ) {
      nused_tot += used;
    }
    LARCV_NORMAL() << "Made " <<  output_cluster_v.size() << " clusters; "
                   << nused_tot << " of " << used_hits_v.size() << " hits used"
                   << std::endl;
          
  }
                                     
  
}
}
