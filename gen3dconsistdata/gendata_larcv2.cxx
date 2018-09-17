#include <iostream>

// larlite
#include "LArUtil/Geometry.h"

// ROOT
#include "TFile.h"
#include "TTree.h"

// NOTE: LARCV2
#incude "larcv/DataFormat/IOManager.h"
#incude "larcv/DataFormat/EventImage2D.h"

int main(int nargs, char** argv ) {

  const larutil::Geometry* geo = larutil::Geometry::GetME();

  larcv::IOManager io(larcv::IOManager::kWRITE );
  io.add_in_file("consistency3d_data.root");
  larcv::ImageMeta y2u_meta( 0, 0, 3456, 2400, 2400, 3456, 0 );
  larcv::ImageMeta y2v_meta( 0, 0, 3456, 2400, 2400, 3456, 1 );  
  larcv::Image2D y2u_intersect_y( y2u_meta );
  larcv::Image2D y2u_intersect_z( y2u_meta );  
  larcv::Image2D y2v_intersect_y( y2v_meta );
  larcv::Image2D y2v_intersect_z( y2v_meta );  

  std::vector<larcv::Image2D> y2u_intersect_v;
  y2u_intersect_v.emplace_back( std::move(y2u_intersect_y) );
  y2u_intersect_v.emplace_back( std::move(y2u_intersect_z) );
  
  std::vector<larcv::Image2D> y2v_intersect_v;
  y2v_intersect_v.emplace_back( std::move(y2v_intersect_y) );
  y2v_intersect_v.emplace_back( std::move(y2v_intersect_z) );
  
  double ypos;
  double zpos;
  for (int y=0; y<3456; y++) {
    source_wire = y;
    for (int puv=0; puv<2; puv++) {
      for (int wire=0; wire<nwires_v[puv]; wire++) {
  	geo->IntersectionPoint( y, wire, 2, puv, ypos, zpos );
	if ( puv==0 ) {
	  y2u_intersect_v[0].set_pixel( wire, y, ypos );
	  y2u_intersect_v[1].set_pixel( wire, y, zpos );
	}
	else {
	  y2v_intersect_v[0].set_pixel( wire, y, ypos );
	  y2v_intersect_v[1].set_pixel( wire, y, zpos );
	}	  
      }
    }
  }

  // save to image2d
  auto ev_y2u = (larcv::EventImage2D*)io.get_data("image2d", "y2u_intersect");
  ev_y2u->emplace( std::move(y2u_intersect_v) );
  auto ev_y2v = (larcv::EventImage2D*)io.get_data("image2d", "y2v_intersect");
  ev_y2v->emplace( std::move(y2u_intersect_v) );

  io.save_entry();

  io.finalize();
  
  return 0;
}
