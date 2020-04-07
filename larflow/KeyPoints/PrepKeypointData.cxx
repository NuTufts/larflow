#include "PrepKeypointData.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

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

  bool PrepKeypointData::_setup_numpy = false;
  
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


    // build key-points
    _kpd_v.clear();
    
    // build crossing points for muon track primaries
    std::vector<KPdata> track_kpd
      = getMuonEndpoints( mcpg, adc_v, mctrack_v, &sce );

    std::cout << "[Track Endpoint Results]" << std::endl;
    for ( auto const& kpd : track_kpd ) {
      std::cout << "  " << str(kpd) << std::endl;
      _kpd_v.emplace_back( std::move(kpd) );
    }

    // add points for shower starts
    std::vector<KPdata> shower_kpd
      = getShowerStarts( mcpg, adc_v, mcshower_v, &sce );
    std::cout << "[Shower Endpoint Results]" << std::endl;
    for ( auto const& kpd : shower_kpd ) {
      std::cout << "  " << str(kpd) << std::endl;
      _kpd_v.emplace_back( std::move(kpd) );      
    }

  }

  
  std::vector<KPdata>
  PrepKeypointData::getMuonEndpoints( ublarcvapp::mctools::MCPixelPGraph& mcpg,
                                      const std::vector<larcv::Image2D>& adc_v,
                                      const larlite::event_mctrack& mctrack_v,
                                      larutil::SpaceChargeMicroBooNE* psce )
  {

    bool verbose = true;
    
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
        kpd.is_shower = 0;
        kpd.origin  = pnode->origin;
        
        std::vector< int > imgcoords_start;
        std::vector< int > imgcoords_end;

        if ( crossingtype>=0 && crossingtype<=2) {
          imgcoords_start
            = ublarcvapp::mctools::CrossingPointsAnaMethods::getFirstStepPosInsideImage( mctrk, adc_v.front().meta(),
                                                                                         4050.0, true, 0.3, 0.1,
                                                                                         kpd.startpt, psce, verbose );
          if ( imgcoords_start.size()>0 ) {
            kpd.imgcoord_start = imgcoords_start;
          }
          
        }

        if ( crossingtype>=0 && crossingtype<=2 ) {
          imgcoords_end
            = ublarcvapp::mctools::CrossingPointsAnaMethods::getFirstStepPosInsideImage( mctrk, adc_v.front().meta(),
                                                                                         4050.0, false, 0.3, 0.1,
                                                                                         kpd.endpt, psce, verbose );
          if ( imgcoords_end.size()>0 ) {
            kpd.imgcoord_end = imgcoords_end;
          }
        }

        kpd_v.emplace_back( std::move(kpd) );
      }

    }//end of primary loop

    return kpd_v;
  }

  std::vector<KPdata>
  PrepKeypointData::getShowerStarts( ublarcvapp::mctools::MCPixelPGraph& mcpg,
                                     const std::vector<larcv::Image2D>& adc_v,
                                     const larlite::event_mcshower& mcshower_v,
                                     larutil::SpaceChargeMicroBooNE* psce )
  {

    // get list of primaries
    std::vector<ublarcvapp::mctools::MCPixelPGraph::Node_t*> primaries
      = mcpg.getPrimaryParticles();

    // output vector of keypoint data
    std::vector<KPdata> kpd_v;

    for ( auto const& pnode : primaries ) {

      if ( abs(pnode->pid)!=11
           && abs(pnode->pid)!=22 )
        continue;

      auto const& mcshr = mcshower_v.at( pnode->vidx );

      //we convert shower to track trajectory
      larlite::mctrack mct;
      mct.push_back( mcshr.Start() );
      mct.push_back( mcshr.End() );

      int crossingtype =
        ublarcvapp::mctools::CrossingPointsAnaMethods::
        doesTrackCrossImageBoundary( mct,
                                     adc_v.front().meta(),
                                     4050.0,
                                     psce );

      if ( crossingtype>=0 ) {

        KPdata kpd;
        kpd.crossingtype = crossingtype;
        kpd.trackid = pnode->tid;
        kpd.pid     = pnode->pid;
        kpd.vid     = pnode->vidx;
        kpd.origin  = pnode->origin;
        
        std::vector< int > imgcoords_start;
        imgcoords_start
          = ublarcvapp::mctools::CrossingPointsAnaMethods::getFirstStepPosInsideImage( mct, adc_v.front().meta(),
                                                                                       4050.0, true, 0.3, 0.1,
                                                                                       kpd.startpt, psce, false );
        if ( imgcoords_start.size()>0 ) {
          kpd.imgcoord_start = imgcoords_start;
          kpd.is_shower = 1;
          kpd_v.emplace_back( std::move(kpd) );
        }
      }

    }//end of primary loop
    
    return kpd_v;
  }

  std::string PrepKeypointData::str( const KPdata& kpd )
  {
    std::stringstream ss;
    ss << "[type=" << kpd.crossingtype << " pid=" << kpd.pid
       << " vid=" << kpd.vid
       << " isshower=" << kpd.is_shower
       << " origin=" << kpd.origin << "] ";

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


  /**
   * return an array with keypoints
   * array columns [tick,wire-U,wire-V,wire-Y,x,y,z,isshower,origin,pid]
   *
   */
  PyObject* PrepKeypointData::get_keypoint_array() const
  {

    if ( !PrepKeypointData::_setup_numpy ) {
      import_array1(0);
      PrepKeypointData::_setup_numpy = true;
    }
    
    // first count the number of unique points
    std::set< std::vector<int> >    unique_coords;
    std::vector< std::vector<int> > kpd_index;
    int npts = 0;
    for ( size_t ikpd=0; ikpd<_kpd_v.size(); ikpd++ ) {
      auto const& kpd = _kpd_v[ikpd];

      if (kpd.imgcoord_start.size()>0) {
        if ( unique_coords.find( kpd.imgcoord_start )==unique_coords.end() ) {
          kpd_index.push_back( std::vector<int>{(int)ikpd,0} );
          unique_coords.insert( kpd.imgcoord_start );
          npts++;
        }
      }
      
      if (kpd.imgcoord_end.size()>0) {
        if ( unique_coords.find( kpd.imgcoord_end )==unique_coords.end() ) {
          kpd_index.push_back( std::vector<int>{(int)ikpd,1} );
          unique_coords.insert( kpd.imgcoord_end );
          npts++;
        }
      }
    }
    
    int nd = 2;
    npy_intp dims[] = { npts, 10 };
    PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNew( nd, dims, NPY_FLOAT );

    size_t ipt = 0;
    for ( auto& kpdidx : kpd_index ) {
      
      auto const& kpd = _kpd_v[kpdidx[0]];

      if ( kpdidx[1]==0 ) {
        // start img coordinates
        for ( size_t i=0; i<4; i++ )
          *((float*)PyArray_GETPTR2(array,ipt,i)) = (float)kpd.imgcoord_start[i];
        // 3D point
        for ( size_t i=0; i<3; i++ )
          *((float*)PyArray_GETPTR2(array,ipt,4+i)) = (float)kpd.startpt[i];
        // is shower
        *((float*)PyArray_GETPTR2(array,ipt,7)) = (float)kpd.is_shower;
        // origin
        *((float*)PyArray_GETPTR2(array,ipt,8)) = (float)kpd.origin;
        // PID
        *((float*)PyArray_GETPTR2(array,ipt,9)) = (float)kpd.pid;
      }
      else if ( kpdidx[1]==1 ) {
        // end img coordinates
        for ( size_t i=0; i<4; i++ )
          *((float*)PyArray_GETPTR2(array,ipt,i)) = (float)kpd.imgcoord_end[i];
        // 3D point
        for ( size_t i=0; i<3; i++ )
          *((float*)PyArray_GETPTR2(array,ipt,4+i)) = (float)kpd.endpt[i];
        // is shower
        *((float*)PyArray_GETPTR2(array,ipt,7)) = (float)kpd.is_shower;
        // origin
        *((float*)PyArray_GETPTR2(array,ipt,8)) = (float)kpd.origin;
        // PID
        *((float*)PyArray_GETPTR2(array,ipt,9)) = (float)kpd.pid;
      }
      ipt++;
    }// end of loop over keypointdata structs

    return (PyObject*)array;
  }
  
}
}
