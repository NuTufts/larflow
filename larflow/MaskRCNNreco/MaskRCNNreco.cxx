#include "MaskRCNNreco.h"

#include "DataFormat/larflow3dhit.h"
#include "DataFormat/larflowcluster.h"
#include "larcv/core/DataFormat/EventClusterMask.h"
#include "larcv/core/DataFormat/EventImage2D.h"

namespace larflow {
namespace mrcnnreco {

  void MaskRCNNreco::process( larcv::IOManager& iolcv,
                              larlite::storage_manager& ioll )
  {

    // get larmatch points
    larlite::event_larflow3dhit* ev_larmatch =
      (larlite::event_larflow3dhit*)ioll.get_data(larlite::data::kLArFlow3DHit,"larmatch");
    LARCV_INFO() << "number of larmatch points: " << (int)ev_larmatch->size() << std::endl;
    
    // get mask-rcnn container
    larcv::EventClusterMask* ev_mrcnn =
      (larcv::EventClusterMask*)iolcv.get_data(larcv::kProductClusterMask,"mask_proposals_y");
    const std::vector<larcv::ClusterMask>& mask_v = ev_mrcnn->as_vector().front();
    LARCV_INFO() << "number of masks: " << mask_v.size() << std::endl;

    // get mask-rcnn container
    larcv::EventImage2D* ev_adc =
      (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D,"wire");
    const std::vector<larcv::Image2D>& adc_v = ev_adc->as_vector();
    
    std::vector<larlite::larflowcluster> cluster_v = clusterbyproposals( *ev_larmatch,
                                                                         mask_v,
                                                                         adc_v,
                                                                         0.5 );

    larlite::event_larflowcluster* ev_out
      = (larlite::event_larflowcluster*)ioll.get_data(larlite::data::kLArFlowCluster,"mrcnn");
    for ( auto& cluster : cluster_v ) {
      ev_out->emplace_back( std::move(cluster) );
    }
  }

  std::vector<larcv::ClusterMask> MaskRCNNreco::merge_proposals( const std::vector<larcv::ClusterMask>& mask_v )
  {
    // N^2 comparison
    // we
    std::vector<larcv::ClusterMask> merged_v;
    return merged_v;
  }

  std::vector<larlite::larflowcluster>
  MaskRCNNreco::clusterbyproposals( const larlite::event_larflow3dhit& ev_larmatch,
                                    const std::vector<larcv::ClusterMask>& mask_v,
                                    const std::vector<larcv::Image2D>& adc_v,
                                    const float hit_threshold )
  {

    std::vector<larlite::larflowcluster> cluster_v;
    auto const& meta = adc_v[2].meta();
    for ( auto& mask : mask_v ) {
      const std::vector<float>& box = mask.as_vector_box_no_convert(); // (colmin,rowmin,colmax,rowmax,class)
      float xmin[2] = { box[0], box[2] };
      float ymin[2] = {  meta.pos_y((int)meta.rows()-box[3]), meta.pos_y((int)meta.rows()-box[1]) };

      LARCV_INFO() << "Finding larmatch points for box: "
                   << "(" << xmin[0] << "," << ymin[0] << "," << xmin[1] << "," << ymin[1] << "," << box[4] << ")"
                   << std::endl;
      
      
      larlite::larflowcluster cluster;
      
      for ( auto& hit : ev_larmatch ) {
        if ( hit[9]<hit_threshold ) continue;
        int yplane_tick = hit.tick;
        int yplane_wire  = hit.targetwire[2];
        if ( yplane_tick>=ymin[0] && yplane_tick<=ymin[1]
             && yplane_wire>=xmin[0] && yplane_wire<=xmin[1] ) {
          cluster.push_back( hit );
        }
      }

      LARCV_INFO() << "assigned " << cluster.size() << " hits to mask." << std::endl;

      cluster_v.push_back(cluster);
    }

    return cluster_v;
  }
}
}
