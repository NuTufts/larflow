#include "PrepDETR2D.h"

#include "larcv/core/DataFormat/EventImage2D.h"
#include "DataFormat/mctrack.h"
#include "DataFormat/mcshower.h"
#include "ublarcvapp/MCTools/NeutrinoVertex.h"

namespace larflow {
namespace detr {

  PrepDETR2D::~PrepDETR2D()
  {
    delete image_v;
    delete bbox_v;
    delete pdg_v;
  }
  
  void PrepDETR2D::setupForOutput()
  {
    _tree = new TTree("detr","Data for DETR training");
    image_v = new std::vector<larcv::NumpyArrayFloat >();
    bbox_v  = new std::vector<larcv::NumpyArrayFloat >();
    pdg_v   = new std::vector<larcv::NumpyArrayInt >();
    _tree->Branch( "image_v", image_v );
    _tree->Branch( "bbox_v",  bbox_v );
    _tree->Branch( "pdg_v",   pdg_v );
  }

  void PrepDETR2D::setupForInput()
  {
    
  }

  void PrepDETR2D::process( larcv::IOManager& iolcv,
                            larlite::storage_manager& ioll )
  {
    // use mcpixelpgraph to define particle instances.
    // make bounding boxes
    // use truth to get the nu vertex in each plane

    // get the truth data we need
    larcv::EventImage2D* ev_adc
      = (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D, "wire" );
    larcv::EventImage2D* ev_instance
      = (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D, "instance" );
    larlite::event_mctrack* ev_mctrack
      = (larlite::event_mctrack*)ioll.get_data(larlite::data::kMCTrack, "mcreco" );
    larlite::event_mcshower* ev_mcshower
      = (larlite::event_mcshower*)ioll.get_data(larlite::data::kMCShower, "mcreco" );

    int nplanes = ev_adc->as_vector().size();

    // build the pixel pgraph
    mcpg.buildgraph( iolcv, ioll );

    // get the true vertex
    std::vector<int> vertex_image_coords = ublarcvapp::mctools::NeutrinoVertex::getImageCoords( ioll );

    // we want to bound the neutrino interaction and the individual particles
    _make_bounding_boxes( iolcv, ioll );
  }

  void PrepDETR2D::_make_bounding_boxes( larcv::IOManager& iolcv, larlite::storage_manager& ioll )
  {
    // use mcpixelpgraph to define particle instances.
    // make bounding boxes
    // use truth to get the nu vertex in each plane

    // get the truth data we need
    larcv::EventImage2D* ev_adc
      = (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D, "wire" );
    larcv::EventImage2D* ev_instance
      = (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D, "instance" );
    larlite::event_mctrack* ev_mctrack
      = (larlite::event_mctrack*)ioll.get_data(larlite::data::kMCTrack, "mcreco" );
    larlite::event_mcshower* ev_mcshower
      = (larlite::event_mcshower*)ioll.get_data(larlite::data::kMCShower, "mcreco" );

    int nplanes = ev_adc->as_vector().size();
    
    nu_bb_v.clear();
    particle_plane_bbox_vv.clear();
    nu_bb_v.resize(nplanes);
        
    for ( auto const& node : mcpg.node_v ) {

      if ( node.origin!=1 )
        continue; // keep neutrino boxes only
      
      int trackid = node.tid;
      int pdg = node.pid;
      std::vector< bbox_t > bb_v;
      bb_v.reserve(nplanes);
      for (int p=0; p<nplanes; p++) {
        auto const& meta = ev_adc->as_vector()[p].meta();
        float mean_wire = node.plane_bbox_twHW_vv[p][1];
        float mean_tick = node.plane_bbox_twHW_vv[p][0];
        float width_wire  = node.plane_bbox_twHW_vv[p][3];
        float height_tick = node.plane_bbox_twHW_vv[p][2];
        bbox_t bb( trackid, pdg );
        bb.min_c = mean_wire - width_wire;
        bb.max_c = mean_wire + width_wire;        
        bb.min_r = mean_tick - height_tick;
        bb.max_r = mean_tick + height_tick;

        // keep within bounds
        if ( bb.min_c<meta.min_x() )
          bb.min_c = meta.min_x();
        if ( bb.max_c>=meta.max_x() )
          bb.max_c = meta.max_x()-0.1;
        
        if ( bb.min_r<meta.min_y() )          
          bb.min_r = meta.min_y();
        if ( bb.max_r>=meta.max_y() )
          bb.max_r = meta.max_y()-0.1;

        // convert to meta row,col
        bb.min_r = meta.row( bb.min_r, __FILE__, __LINE__ );
        bb.max_r = meta.row( bb.max_r, __FILE__, __LINE__ );
        bb.min_c = meta.col( bb.min_c, __FILE__, __LINE__ );
        bb.max_c = meta.col( bb.max_c, __FILE__, __LINE__ );

        // update nu interaction bbox
        nu_bb_v[p].update( bb.min_c, bb.min_r );
        nu_bb_v[p].update( bb.max_c, bb.max_r );
        
        bb_v.push_back( bb );
      }//loop over planes

      particle_plane_bbox_vv.emplace_back( std::move(bb_v) );
    }//loop over all nodes

    LARCV_NORMAL() << "defined " << particle_plane_bbox_vv.size() << " particle bounding boxes" << std::endl;
  }

  void PrepDETR2D::_make_numpy_arrays()
  {
    // first we make the whole image crop
    // then we adjust the bounding boxes to fit the crop
    
  }
   
  
}
}
