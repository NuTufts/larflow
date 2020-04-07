#include "PrepKeypointData.h"

#include <sstream>

#include "ublarcvapp/MCTools/MCPixelPGraph.h"
#include "ublarcvapp/MCTools/crossingPointsAnaMethods.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/Image2D.h"
#include "larcv/core/DataFormat/EventImage2D.h"

// larlite
#include "LArUtil/SpaceChargeMicroBooNE.h"
#include "DataFormat/storage_manager.h"
#include "DataFormat/mctrack.h"
#include "DataFormat/mcshower.h"
#include "DataFormat/mctruth.h"

namespace larflow {
namespace keypoints {

  /**
   * process one event, given io managers
   *
   */
  void PrepKeypointData::process( larcv::IOManager& iolcv, larlite::storage_manager& ioll )
  {
    auto ev_adc      = (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D,"wiremc");
    auto ev_segment  = (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D,"segment");
    auto ev_instance = (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D,"instance");
    auto ev_ancestor = (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D,"ancestor");

    auto ev_mctrack  = (larlite::event_mctrack*)ioll.get_data(  larlite::data::kMCTrack, "mcreco" );
    auto ev_mcshower = (larlite::event_mcshower*)ioll.get_data( larlite::data::kMCShower, "mcreco" );
    auto ev_mctruth  = (larlite::event_mctruth*)ioll.get_data(  larlite::data::kMCTruth, "generator" );
    
    std::vector<larcv::Image2D> badch_v;
    for ( auto const& img : ev_adc->Image2DArray() ) {
      larcv::Image2D blank(img.meta());
      blank.paint(0.0);
      badch_v.emplace_back( std::move(blank) );
    }

    std::cout << "[PrepKeypointData Inputs]" << std::endl;
    std::cout << "  adc images: "      << ev_adc->Image2DArray().size() << std::endl;
    std::cout << "  badch images: "    << badch_v.size() << std::endl;    
    std::cout << "  segment images: "  << ev_segment->Image2DArray().size() << std::endl;
    std::cout << "  instance images: " << ev_instance->Image2DArray().size() << std::endl;
    std::cout << "  ancestor images: " << ev_ancestor->Image2DArray().size() << std::endl;
    std::cout << "  mctracks: " << ev_mctrack->size() << std::endl;
    std::cout << "  mcshowers: " << ev_mcshower->size() << std::endl;
    std::cout << "  mctruths: " << ev_mctruth->size() << std::endl;
                                                         
    
    process( ev_adc->Image2DArray(),
             badch_v,
             ev_segment->Image2DArray(),
             ev_instance->Image2DArray(),
             ev_ancestor->Image2DArray(),
             *ev_mctrack,
             *ev_mcshower,
             *ev_mctruth );
  }

  /**
   * process one event, directly given input containers
   *
   */
  void PrepKeypointData::process( const std::vector<larcv::Image2D>&    adc_v,
                                  const std::vector<larcv::Image2D>&    badch_v,
                                  const std::vector<larcv::Image2D>&    segment_v,
                                  const std::vector<larcv::Image2D>&    instance_v,
                                  const std::vector<larcv::Image2D>&    ancestor_v,                                  
                                  const larlite::event_mctrack&  mctrack_v,
                                  const larlite::event_mcshower& mcshower_v,
                                  const larlite::event_mctruth&  mctruth_v ) {

    // allocate space charge class
    larutil::SpaceChargeMicroBooNE sce;

    // make particle graph
    ublarcvapp::mctools::MCPixelPGraph mcpg;
    mcpg.buildgraph( adc_v, segment_v, instance_v, ancestor_v,
                     mcshower_v, mctrack_v, mctruth_v );


    // build crossing points for muon track primaries
    std::vector<PrepKeypointData::KPdata> track_kpd
      = getMuonEndpoints( mcpg, adc_v, mctrack_v, &sce );

    std::cout << "[Track Endpoint Results]" << std::endl;
    for ( auto const& kpd : track_kpd ) {
      std::cout << "  " << str(kpd) << std::endl;
    }

  }

  
  std::vector<PrepKeypointData::KPdata>
  PrepKeypointData::getMuonEndpoints( ublarcvapp::mctools::MCPixelPGraph& mcpg,
                                      const std::vector<larcv::Image2D>& adc_v,
                                      const larlite::event_mctrack& mctrack_v,
                                      larutil::SpaceChargeMicroBooNE* psce )
  {

    // get list of primaries
    std::vector<ublarcvapp::mctools::MCPixelPGraph::Node_t*> primaries
      = mcpg.getPrimaryParticles();

    // output vector of keypoint data
    std::vector<KPdata> kpd_v;

    for ( auto const& pnode : primaries ) {

      if ( abs(pnode->pid)!=13
           && abs(pnode->pid)!=2212
           && abs(pnode->pid)!=211 )
        continue;

      auto const& mctrk = mctrack_v.at( pnode->vidx );

      int crossingtype =
        ublarcvapp::mctools::CrossingPointsAnaMethods::
        doesTrackCrossImageBoundary( mctrk,
                                     adc_v.front().meta(),
                                     4050.0,
                                     psce );

      if ( crossingtype>=0 ) {

        KPdata kpd;
        kpd.crossingtype = crossingtype;
        kpd.trackid = pnode->tid;
        kpd.pid     = pnode->pid;
        kpd.vid     = pnode->vidx;
        
        std::vector< int > imgcoords_start;
        std::vector< int > imgcoords_end;

        if ( crossingtype==0 || crossingtype==2) {
          imgcoords_start
            = ublarcvapp::mctools::CrossingPointsAnaMethods::getFirstStepPosInsideImage( mctrk, adc_v.front().meta(),
                                                                                         4050.0, true, 0.3, 0.1,
                                                                                         kpd.startpt, psce );
          if ( imgcoords_start.size()>0 ) {
            kpd.imgcoord_start = imgcoords_start;
          }
          
        }

        if ( crossingtype==1 || crossingtype==2 ) {
          imgcoords_end
            = ublarcvapp::mctools::CrossingPointsAnaMethods::getFirstStepPosInsideImage( mctrk, adc_v.front().meta(),
                                                                                         4050.0, false, 0.3, 0.1,
                                                                                         kpd.endpt, psce );
          if ( imgcoords_end.size()>0 ) {
            kpd.imgcoord_end = imgcoords_end;
          }
        }

        kpd_v.emplace_back( std::move(kpd) );
      }

    }//end of primary loop

    return kpd_v;
  }

  std::string PrepKeypointData::str( const PrepKeypointData::KPdata& kpd )
  {
    std::stringstream ss;
    ss << "[type=" << kpd.crossingtype << " pid=" << kpd.pid << " vid=" << kpd.vid << "] ";

    if ( kpd.imgcoord_start.size()>0 )
      ss << " imgstart=(" << kpd.imgcoord_start[0] << ","
         << kpd.imgcoord_start[1] << ","
         << kpd.imgcoord_start[2] << ","
         << kpd.imgcoord_start[3] << ") ";

    if ( kpd.startpt.size()>0 )
      ss << " startpt=(" << kpd.startpt[0] << "," << kpd.startpt[1] << "," << kpd.startpt[2] << ") ";
        
    if ( kpd.imgcoord_end.size()>0 ) 
      ss << " imgend=(" << kpd.imgcoord_end[0] << ","
         << kpd.imgcoord_end[1] << ","
         << kpd.imgcoord_end[2] << ","
         << kpd.imgcoord_end[3] << ") ";

    if ( kpd.endpt.size()>0 )
      ss << " endpt=(" << kpd.endpt[0] << "," << kpd.endpt[1] << "," << kpd.endpt[2] << ") ";
    
    return ss.str();
  }
  
}
}
