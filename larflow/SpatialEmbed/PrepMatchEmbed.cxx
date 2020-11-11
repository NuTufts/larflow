#include "PrepMatchEmbed.h"

#include <map>

#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/Image2D.h"
#include "larcv/core/DataFormat/EventImage2D.h"
#include "DataFormat/storage_manager.h"

namespace larflow {
namespace spatialembed {

  /**
   *
   * prepare truth labels for the spatial embedding clustering task
   *
   * @param[in] iolcv    larcv::IOManager instance containing needed data products.
   * @param[in] ioll     larlite::storage_manager instance containing needed data products.
   * @param[in] triplets instance containing proposed pixel triplets
   * 
   */
  void PrepMatchEmbed::process( larcv::IOManager& iolcv,
                                larlite::storage_manager& ioll,
                                const larflow::prep::PrepMatchTriplets& triplets )
  {

    larcv::EventImage2D* ev_image
      = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, "wiremc" );
    
    larcv::EventImage2D* ev_ancestor
      = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, "ancestor" );

    // collect 2D pixel clusters based on ancestor ID number
    for (size_t p=0; p<3; p++ )
      _collect_ancestor_pixels( ev_image->Image2DArray()[p], ev_ancestor->Image2DArray()[p] );

    // assign ancestor ID to triplets
    _assign_triplet_ancestor_id( triplets, ev_ancestor->Image2DArray() );
    
  }
  
  /** 
   * scan image and collect ancestor ID pixels
   */
  void PrepMatchEmbed::_collect_ancestor_pixels( const larcv::Image2D& adc,
                                                 const larcv::Image2D& ancestor )
  {

    int plane = (int)adc.meta().plane();
    pix_map_m[plane].clear();

    auto& pixmap = pix_map_m[plane];

    auto const& meta = ancestor.meta();
    for (int r=0; r<(int)meta.rows(); r++) {
      for (int c=0; c<(int)meta.cols(); c++) {
        
        float pixval = adc.pixel(r,c);
        if ( pixval>10.0 ) {
          int id = (int)ancestor.pixel(r,c);
          if ( pixmap.find(id)==pixmap.end() ) {
            pixmap[ id ] = std::vector<AncestorIDPix_t>();
          }


          AncestorIDPix_t pix( r, c, id );
          pixmap[id].push_back( pix );

        }
      }
    }

    std::cout << "[ PrepMatchEmbed::_collect_ancestor_stats ] collect ancestor instances" << std::endl;
    std::cout << "  plane: " << plane << std::endl;
    std::cout << "  number of IDs: " << pixmap.size() << std::endl;
    int i=0;
    for (auto it=pixmap.begin(); it!=pixmap.end(); it++) {
      std::cout << "  [" << i << "] id=" << it->first << " npixels=" << it->second.size() << std::endl;
      // sort for easier search
      std::sort( it->second.begin(), it->second.end() );
    }
    
  }


  const std::vector<AncestorIDPix_t>& PrepMatchEmbed::get_instance_pixlist(int plane, int aid )
  {

    auto it=pix_map_m[plane].find(aid);
    if ( it==pix_map_m[plane].end() ) {
      throw std::runtime_error("no ancestor id in pix_map_m");
    }
    return it->second;
    
  }

  /*
   * assign an ancestor id label to each triplet. 
   * also collect indices to triplets for those with the same ancestor id.
   * in effect clusters the triplet/3d spacepoints based on truth information found in ancestor id images.
   *
   */
  void PrepMatchEmbed::_assign_triplet_ancestor_id( const larflow::prep::PrepMatchTriplets& triplet,
                                                    const std::vector<larcv::Image2D>& ancestor_v )
  {
    
  }
}
}
