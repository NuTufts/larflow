#include "PrepKeypointData.h"

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
    //larutil::SpaceChargeMicroBooNE sce;

    // make particle graph
    ublarcvapp::mctools::MCPixelPGraph mcpg;
    mcpg.buildgraph( adc_v, segment_v, instance_v, ancestor_v,
                     mcshower_v, mctrack_v, mctruth_v );


    // build crossing points for muon track primaries

  }

  
  void PrepKeypointData::getMuonEndpoints( ublarcvapp::mctools::MCPixelPGraph& mcpg,
                                           const std::vector<larcv::Image2D>& adc_v,
                                           const larlite::event_mctrack& mctrack_v,
                                           larutil::SpaceChargeMicroBooNE* psce )
  {

    // get list of primaries
    std::vector<ublarcvapp::mctools::MCPixelPGraph::Node_t*> primaries
      = mcpg.getPrimaryParticles();

    for ( auto const& pnode : primaries ) {
      if ( pnode->pid!=13 || pnode->pid!=-13 ) continue; // skip non-muons

      auto const& mctrk = mctrack_v.at( pnode->vidx );

      int crossingtype =
        ublarcvapp::mctools::CrossingPointsAnaMethods::
        doesTrackCrossImageBoundary( mctrk,
                                     adc_v.front().meta(),
                                     4050.0,
                                     psce );

      if ( crossingtype<0 ) continue;
      
      std::vector< int > imgcoords_start
        = ublarcvapp::mctools::CrossingPointsAnaMethods::getFirstStepPosInsideImage( mctrk, adc_v.front().meta(),
                                                                                     4050.0, true, 0.3, 0.1, psce );
      std::vector< int > imgcoords_end
        = ublarcvapp::mctools::CrossingPointsAnaMethods::getFirstStepPosInsideImage( mctrk, adc_v.front().meta(),
                                                                                     4050.0, false, 0.3, 0.1, psce );
      
    }
                                                                                     
    
  }
  
}
}
