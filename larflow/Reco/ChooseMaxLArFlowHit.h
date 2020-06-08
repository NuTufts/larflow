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
#include "larcv/core/DataFormat/EventImage2D.h"

#include "DataFormat/storage_manager.h"
#include "DataFormat/larflow3dhit.h"

namespace larflow {
namespace reco {

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
    
    void _make_pixelmap( const larlite::event_larflow3dhit& hit_v,
                         const std::vector<larcv::Image2D>& img_v,
                         const int source_plane,
                         std::vector<int>& idx_used_v );
    
    std::map< Pixel_t, std::vector<int> > _srcpixel_to_spacepoint_m; ///< map source image pixel to space point triplet

  protected:

    std::string _input_larflow3dhit_treename;
    std::string _output_larflow3dhit_treename;

  public:

    void set_input_larflow3dhit_treename( std::string name )  { _input_larflow3dhit_treename=name; };
    void set_output_larflow3dhit_treename( std::string name ) { _output_larflow3dhit_treename=name; };

  };


}
}

#endif
