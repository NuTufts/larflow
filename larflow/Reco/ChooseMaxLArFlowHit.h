#ifndef __CHOOSE_MAX_LARFLOW_HIT_H__
#define __CHOOSE_MAX_LARFLOW_HIT_H__

/**
 * We reduce the number of larflow hits, by choosing the maximum flow 
 * from a source pixel
 *
 */

#include <string>
#include <map>

#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "DataFormat/storage_manager.h"

namespace larflow {
namespace reco {

  class ChooseMaxLArFlowHit : public larcv::larcv_base {

  public:

    ChooseMaxLArFlowHit()
      : larcv::larcv_base("ChooseMaxLArFlowHit")
      {};

    virtual ~ChooseMaxLArFlowHit() {};

    void process( larcv::IOManager& iolcv, larlite::storage_manager& ioll );

  public:
    struct Pixel_t {
      int row;
      int col;
      int plane;
      bool operator<( const Pixel_t& rhs ) const {
        if ( plane<rhs.plane ) return true;
        else if ( plane==rhs.plane && row<rhs.row ) return true;
        else if ( plane==rhs.plane && row==rhs.row && col<rhs.col ) return true;
        return false;
      };
    };

  protected:
    
    void _make_pixelmap() {};
    std::map< Pixel_t, int > _srcpixel_to_spacepoint_m; ///< map source image pixel to space point triplet

  protected:

    std::string _input_larflow3dhit_tree;

  public:

    void set_input_larflow3dhit_treename( std::string name ) { _input_larflow3dhit_tree=name; };

  };


}
}

#endif
