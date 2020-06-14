#include <iostream>

// larlite
#include "LArUtil/Geometry.h"

// ROOT
#include "TFile.h"
#include "TTree.h"

// NOTE: LARCV1
#include "DataFormat/IOManager.h"
#include "DataFormat/EventImage2D.h"
#include "DataFormat/Image2D.h"
#include "DataFormat/ImageMeta.h"

int main(int nargs, char** argv ) {

  const larutil::Geometry* geo = larutil::Geometry::GetME();

  larcv::IOManager io(larcv::IOManager::kWRITE );
  io.set_out_file("consistency3d_data_larcv1.root");
  io.initialize();
  larcv::ImageMeta y2u_meta( 3456, 2400, 2400, 3456, 0, 0, 0 );
  larcv::ImageMeta y2v_meta( 3456, 2400, 2400, 3456, 0, 0, 1 );
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

  std::vector<int> nwires_v = {2400,2400,3456};
  
  double ypos;
  double zpos;
  for (int y=0; y<nwires_v[2]; y++) {
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
  io.set_id(0,0,0);
  auto ev_y2u = (larcv::EventImage2D*)io.get_data(larcv::kProductImage2D, "y2u_intersect");
  ev_y2u->Emplace( std::move(y2u_intersect_v) );
  auto ev_y2v = (larcv::EventImage2D*)io.get_data(larcv::kProductImage2D, "y2v_intersect");
  ev_y2v->Emplace( std::move(y2v_intersect_v) );

  io.save_entry();

  io.finalize();
  
  return 0;
}
