#ifndef __CHOOSE_MAX_LARFLOW_HIT_H__
#define __CHOOSE_MAX_LARFLOW_HIT_H__


#include <string>
#include <map>

#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/EventImage2D.h"

#include "larlite/DataFormat/storage_manager.h"
#include "larlite/DataFormat/larflow3dhit.h"

namespace larflow {
namespace reco {
  
  /**
   * @ingroup Reco
   * @class ChooseMaxLArFlowHit
   * @brief reduce the number of larflow hits by choosing the maximum flow from a source pixel
   *
   * When generating possible space points, a pixel in a given image can be a part
   * of several possible space points.
   *
   * This class reduces the number of hits such that only one space point
   * exists per pixel per row (i.e. time bin) per plane on a given image.
   * This reduction is done for each plane. We take the union of reduced hits
   * from the three planes.
   *
   * Example on one plane: a list of space points might have the following image coordinates
   * given as (row, colU, colV, colY, larmatch score):
   * (row=10, 1, 2, 3, 0.8 ), (row=15,5, 6, 3, 0.3), (row=15, 8, 1, 3, 0.9).
   * This class, if operating only on the Y-plane, reduces this list to:
   * (row=10, 1, 2, 3, 0.8 ), (row=15, 8, 1, 3, 0.9).
   *
   *
   */  
  class ChooseMaxLArFlowHit : public larcv::larcv_base {

  public:

    ChooseMaxLArFlowHit()
      : larcv::larcv_base("ChooseMaxLArFlowHit"),
      _input_larflow3dhit_treename("larmatch"),
      _output_larflow3dhit_treename("maxlarmatch")
      {};

    virtual ~ChooseMaxLArFlowHit() {};

    void process( larcv::IOManager& iolcv, larlite::storage_manager& ioll );

  public:

    /**
     * @struct Pixel_t
     * @brief Internal struct representing a pixel in wire plane image
     */
    struct Pixel_t {
      int row;   ///< row in 2D image
      int col;   ///< col in 2D image
      int plane; ///< plane index of image
      int tpc;
      int cryo;

      /**
       * @brief comparator ordering by (plane,row,col)
       */
      bool operator<( const Pixel_t& rhs ) const {
        if ( plane<rhs.plane ) return true;
        else if ( plane==rhs.plane && row<rhs.row ) return true;
        else if ( plane==rhs.plane && row==rhs.row && col<rhs.col ) return true;
        return false;
      };
    };

  protected:
    
    void _make_pixelmap( const larlite::event_larflow3dhit& hit_v,
                         const larcv::Image2D& img,
                         const int source_plane,
			 const int tpcid,
			 const int cryoid,
                         std::vector<int>& idx_used_v );
    
    std::map< Pixel_t, std::vector<int> > _srcpixel_to_spacepoint_m; ///< map source image pixel to space point triplet

  protected:

    std::string _input_larflow3dhit_treename;  ///< name of tree to get hits from
    std::string _output_larflow3dhit_treename; ///< name of tree to store hits in

  public:

    /** 
     * @brief set the input tree name to get hits from
     * @param[in]  name Set name of input hit tree. Tree follows naming pattern image2d_[name]_tree.
     */
    void set_input_larflow3dhit_treename( std::string name )  { _input_larflow3dhit_treename=name; };

    /** 
     * @brief set the output tree name to get hits from
     * @param[in]  name Set name of output hit tree. Tree follows naming pattern image2d_[name]_tree.
     */    
    void set_output_larflow3dhit_treename( std::string name ) { _output_larflow3dhit_treename=name; };

  };


}
}

#endif
