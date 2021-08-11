#ifndef __LARFLOW_DETR_PREP_DETR_2D_H__
#define __LARFLOW_DETR_PREP_DETR_2D_H__

#include <vector>
#include "TTree.h"
#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/PyUtil/NumpyArray.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "DataFormat/storage_manager.h"
#include "ublarcvapp/MCTools/MCPixelPGraph.h"

namespace larflow {
namespace detr {

  /**
   * @class PrepDETR2D
   * @brief prepare 2D instance segmentation data
   *
   * We crop around the true vertex. We define bounding boxes. 
   * 
   * From the DETR source. What the bounding box targets should look like:
   * Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
   *  targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
   *  The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
   */
  class PrepDETR2D : public larcv::larcv_base {
    
  public:

  PrepDETR2D()
    : larcv::larcv_base("PrepDETR2D"),
      _width(0),
      _height(0),
      image_v(nullptr),
      bbox_v(nullptr),
      pdg_v(nullptr)
      {};
    
  PrepDETR2D( int img_width_pixels, int img_height_pixels, int plane )
    : larcv::larcv_base("PrepDETR2D"),
      _width(img_width_pixels),
      _height(img_height_pixels),
      image_v(nullptr),
      bbox_v(nullptr),
      pdg_v(nullptr)
      {};
    virtual ~PrepDETR2D();
    
    void process( larcv::IOManager& iolcv, larlite::storage_manager& ioll );
    void setupForOutput();
    void setupForInput();
    void setCropSize( int w ,int h ) { _width=w; _height=h; };
    void clear();


    int _width;
    int _height;
    
    TTree* _tree;
    std::vector< larcv::NumpyArrayFloat >* image_v; ///< image crop
    std::vector< larcv::NumpyArrayFloat >* bbox_v;  ///< bounding box annotations for image (x,y,w,h)
    std::vector< larcv::NumpyArrayInt >*   pdg_v;   ///< particle class for each bounding box

    ublarcvapp::mctools::MCPixelPGraph mcpg;

    struct bbox_t {
      float min_c;
      float min_r;
      float max_c;
      float max_r;
      int trackid;
      int pdg;
      bbox_t(int tid=0, int pid=0)
        : min_c(1.0e9),
          min_r(1.0e9),
          max_c(0),
          max_r(0),
          trackid(tid),
          pdg(pid)
      {};
      void update( float c, float r )
      {
        if ( min_c>c ) min_c = c;
        if ( min_r>r ) min_r = r;
        if ( max_c<c ) max_c = c;
        if ( max_r<r ) max_r = r;
      };
    };
        
    std::vector<bbox_t> nu_bb_v;
    std::vector< std::vector<bbox_t> > particle_plane_bbox_vv;
    std::vector< std::vector<bbox_t> > plane_subshower_bbox_vv;
    
    void _make_bounding_boxes( larcv::IOManager& iolcv, larlite::storage_manager& ioll );

    void _subclusterShowers( const std::vector<larcv::Image2D>& adc_v,
                             ublarcvapp::mctools::MCPixelPGraph& mcpg );
    
    void _make_numpy_arrays( larcv::IOManager& iolcv,
                             larlite::storage_manager& ioll );
    
  };
  
}
}

#endif
