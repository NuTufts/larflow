#include "PrepDETR2D.h"

#include "larcv/core/DataFormat/EventImage2D.h"
#include "DataFormat/mctrack.h"
#include "DataFormat/mcshower.h"
#include "ublarcvapp/MCTools/NeutrinoVertex.h"

#include "larflow/Reco/cluster_functions.h"

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

  void PrepDETR2D::clear()
  {
    if ( !image_v || !bbox_v || !pdg_v ) {
      throw std::runtime_error("Need to setup class before using. Call setupForOutput() or setupForInput().");
    }
    image_v->clear();
    bbox_v->clear();
    pdg_v->clear();
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
    mcpg.printGraph();

    // we want to bound the neutrino interaction and the individual particles
    _make_bounding_boxes( iolcv, ioll );
    _subclusterShowers( ev_adc->as_vector(), mcpg );
    
    // turn image and bbox data into larcv::NumpyArray
    _make_numpy_arrays( iolcv, ioll );
    
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
    particle_plane_bbox_vv.resize( nplanes );
        
    for ( auto const& node : mcpg.node_v ) {

      if ( node.origin!=1 )
        continue; // keep neutrino boxes only

      if ( abs(node.pid)==11 || abs(node.pid)==22 )
        continue; // skip showers
      
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

        if ( width_wire>0 && height_tick>0 ) {

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

          particle_plane_bbox_vv[p].emplace_back( std::move(bb) );
          
        }//end of if valid bbox
        //bb_v.push_back( bb );
      }//loop over planes

      //particle_plane_bbox_vv.emplace_back( std::move(bb_v) );
    }//loop over all nodes

    LARCV_NORMAL() << "defined " << particle_plane_bbox_vv.size() << " particle bounding boxes" << std::endl;
  }

  void PrepDETR2D::_make_numpy_arrays( larcv::IOManager& iolcv,
                                       larlite::storage_manager& ioll )
  {

    LARCV_DEBUG() << "Start" << std::endl;

    // check the crop width and height
    if ( _height==0 || _width==0 ) {
      throw std::runtime_error("Need to call setCropSize(int width, int height) before making numpy arrays");
    }
    
    clear();
    
    // first we make the image crop
    larcv::EventImage2D* ev_adc
      = (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D, "wire" );
    auto const& adc_v = ev_adc->as_vector();

    LARCV_DEBUG() << "number of planes: " << adc_v.size() << std::endl;
      
    // get the true vertex tick and wires
    std::vector<int> vertex_image_coords = ublarcvapp::mctools::NeutrinoVertex::getImageCoords( ioll );

    LARCV_DEBUG() << "nu vertex @ img coord = ("
                  << vertex_image_coords[0] << ", "
                  << vertex_image_coords[1] << ", "
                  << vertex_image_coords[2] << ", "
                  << vertex_image_coords[3] << ")"
                  << std::endl;

    for (size_t p=0; p<adc_v.size(); p++) {

      larcv::NumpyArrayFloat crop;
      
      auto const& meta = adc_v[p].meta();      
    
      int original_width  = meta.cols();
      int original_height = meta.rows();

      int crop_center_wire = vertex_image_coords[p];
      int crop_center_tick = vertex_image_coords[3];
      
      if ( crop_center_wire <= meta.min_x()
           || crop_center_wire>=meta.max_x()
           || crop_center_tick <= meta.min_y()
           || crop_center_tick>=meta.max_y() ) {
        LARCV_DEBUG() << "plane " << p << " has out-of-bounds vertex" << std::endl;
        image_v->emplace_back( std::move(crop) );
        continue;
      }
      
      int crop_center_col = meta.col( crop_center_wire );
      int crop_center_row = meta.row( crop_center_tick );

      // define the crop bounds within original image pixel coordinates
      int min_col = crop_center_col - 0.5*_width;
      int max_col = min_col + _width - 1; // inclusive bounds
      int min_row = crop_center_row - 0.5*_height;
      int max_row = min_row + _height - 1; // inclusive bounds

      // range of pixel values to copy from original
      int start_col = ( min_col<0 ) ? 0 : min_col;
      int stop_col  = ( max_col>=(int)meta.cols()-1 ) ? (int)meta.cols()-1 : max_col;
      int start_row = ( min_row<0 ) ? 0 : min_row;
      int stop_row  = ( max_row>=(int)meta.rows()-1 ) ? (int)meta.rows()-1 : max_row;
      LARCV_DEBUG() << "origin(r,c)=(" << min_row << "," << min_col << ")" << std::endl;
      LARCV_DEBUG() << "copy row=[" << start_row << "," << stop_row << "] col=[" << start_col << "," << stop_col << "]" << std::endl;

      auto const& data = adc_v[p].as_vector();

      // copy into larcv::NumpyArray object
      crop.ndims = 2;
      crop.shape.resize(2,0);
      crop.shape[0] = _height;
      crop.shape[1] = _width;
      
      LARCV_DEBUG() << "allocate crop: " << _width*_height << std::endl;
      
      crop.data.resize( _width*_height, 0.0 );
      
      for( int c=start_col; c<=stop_col; c++ ) {
        int dc = c-min_col;
        for (int r=start_row; r<=stop_row; r++) {
          int dr = r-min_row;
          // LARCV_DEBUG() << "  copy (" << dr << "," << dc << ", idx=" << _height*dc+dr << ") from "
          //               << " orig(" << r << "," << c << ", idx=" << c*meta.rows()+r << ")" << std::endl;
          crop.data[ _height*dc + dr ] = data[ c*meta.rows() + r ];
        }
      }

      LARCV_DEBUG() << "store crop" << std::endl;
      image_v->emplace_back( std::move(crop) );

      // now package the bounding boxes
      std::vector< std::vector<float> > bb_v;
      //for ( auto const& particle_bbox : particle_plane_bbox_vv ) {
      for ( auto const& plane_bbox : particle_plane_bbox_vv[p] ) {      
        
        float c1 = (plane_bbox.min_c-(float)min_col)/(float)_width;
        float c2 = (plane_bbox.max_c-(float)min_col)/(float)_width;
        
        float r1 = (plane_bbox.min_r-(float)min_row)/(float)_height;
        float r2 = (plane_bbox.max_r-(float)min_row)/(float)_height;

        if ( (r1<=0.0 && r2<=0.0)
             || (r1>=1.0 && r2>=1.0 )
             || (c1<=0.0 && c2<=0.0)
             || (c1>=1.0 && c2>=1.0)
             || (r1<=0 && r2>=0 && c1<=0 & & c2>=0) ) {
          // out of bbox
          continue; 
        }
        
        if ( fabs(r2-r1)<2.0/_height || fabs(c2-c1)<2.0/_width ) {
          // too narrow or short
          continue;
        }

        LARCV_DEBUG() << "  plane[" << p << "] normalized bbox: (" << r1 << "," << c1 << ") to (" << r2 << "," << c2  << ")"
                      << " pdg=" << plane_bbox.pdg << " trackid=" << plane_bbox.trackid
                      << std::endl;
        
        if ( c1<0 )   c1 = 0.;
        if ( c1>1.0 ) c1 = 1.0;

        if ( c2<0 )   c2 = 0.;
        if ( c2>1.0 ) c2 = 1.;
        
        if ( r1<0 ) r1 = 0.;
        if ( r1>1 ) r1 = 1.0;

        if ( r2<0 ) r2 = 0.0;
        if ( r2>1 ) r2 = 1.0;

        std::vector<float> bb(5);
        bb[0] = 0.5*(r1+r2);
        bb[1] = 0.5*(c1+c2);
        bb[2] = 0.5*fabs(r2-r1);
        bb[3] = 0.5*fabs(c2-c1);
        bb[4] = plane_bbox.pdg;
        bb[5] = p;

        bb_v.push_back( bb );      

        
      }//particle loop

      larcv::NumpyArrayFloat np_bb;
      if ( bb_v.size()==0 ) {
        //empty
        bbox_v->emplace_back( std::move(np_bb) );
        continue;
      }

      LARCV_DEBUG() << "Storing " << bb_v.size() << " bounding boxes for plane " << p << std::endl;
      
      np_bb.ndims = 2;
      np_bb.shape.resize(2,0);
      np_bb.shape[0] = bb_v.size();
      np_bb.shape[1] = 6;
      np_bb.data.resize(6*bb_v.size());      
      for (int i=0; i<(int)bb_v.size(); i++) {
        for (int j=0; j<6; j++)
          np_bb.data[ 6*i + j ] = bb_v[i][j];
      }
      
      bbox_v->emplace_back( std::move(np_bb) );
    }//end of plane loop
    
    LARCV_DEBUG() << "check num images stored" << std::endl;
    
    if ( image_v->size()!=adc_v.size() ) {
      throw std::runtime_error("Number of crops not the same as the number of original images");
    }
    if (bbox_v->size()!=adc_v.size() ) {
      throw std::runtime_error("Number of bounding box arrays not the same as the number of original images");
    }
    LARCV_DEBUG() << "finished" << std::endl;    
  }

  /**
   * @brief Use dbscan to subcluster shower pixels
   *
   */
  void PrepDETR2D::_subclusterShowers( const std::vector<larcv::Image2D>& adc_v,
                                       ublarcvapp::mctools::MCPixelPGraph& mcpg )
  {

    plane_subshower_bbox_vv.clear();
    plane_subshower_bbox_vv.resize( adc_v.size() );
    
    for ( auto& node : mcpg.node_v ) {
      if (node.origin!=1 ) continue; // skip non-neutrino
      if (abs(node.pid)==11 || abs(node.pid)==22) {

        LARCV_DEBUG() << "cluster shower node, trackid[" << node.tid << "]" << std::endl;
        
        // must be shower
        size_t nplanes = node.pix_vv.size();
        
        for (int p=0; p<nplanes; p++) {
          auto const& meta = adc_v[p].meta();
          LARCV_DEBUG() << " plane " << p << " npix=" << node.pix_vv[p].size() << std::endl;
          if ( node.pix_vv[p].size()<2*10 ) continue;
          
          std::vector< std::vector<float> > hit_v;
          hit_v.reserve( node.pix_vv[p].size() );
          int npix = node.pix_vv[p].size()/2;
          for (int ipix=0; ipix<npix; ipix++){

            std::vector<float> xyz(3,0);
            // pixel location given in tick and wire coordinates
            float tick = node.pix_vv[p][ipix*2];
            float wire = node.pix_vv[p][ipix*2+1];
            if ( tick<=meta.min_y() || tick>=meta.max_y() )
              continue;
            if ( wire<=meta.min_x() || wire>=meta.max_x() )
              continue;
            xyz[1] = meta.row(tick,__FILE__,__LINE__);   // wire: 1 wire per pixel
            xyz[0] = meta.col(wire,__FILE__,__LINE__); // tick: 6 ticks per pixel
            float adc = adc_v[p].pixel( (int)xyz[1], (int)xyz[0] );
            if ( adc>10.0 )
              hit_v.push_back( xyz );
          }
          std::vector< larflow::reco::cluster_t > cluster_v;
          larflow::reco::cluster_sdbscan_spacepoints( hit_v, cluster_v, 5.0,  10, 20 );
          
          for ( auto& cluster : cluster_v ) {
            // define a new bounding box for each subshower fragment
            bbox_t subshower( node.tid, node.pid );
            float pixsum = 0.;
            for ( auto& hit : cluster.points_v ) {
              subshower.update( hit[0], hit[1] );
              pixsum += adc_v[p].pixel( (int)hit[1], (int)hit[0] );              
            }
            //plane_subshower_bbox_vv[p].push_back( subshower );
            float w = fabs(subshower.max_c-subshower.min_c);
            float h = fabs(subshower.max_r-subshower.max_r);
            if ( pixsum>1000.0 && (w>=10 || h>=10) )
              particle_plane_bbox_vv[p].push_back( subshower );
          }
          LARCV_DEBUG() << "shower id[" << node.tid << "] plane[" << p << "] broke into " << cluster_v.size() << " subshowers" << std::endl;
        }//end of pane loop     
      }//end of if shower
    }//end of node loop
    
  }
  
}
}
