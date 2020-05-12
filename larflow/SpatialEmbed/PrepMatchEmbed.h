#ifndef __PrepMatchEmbed_h__
#define __PrepMatchEmbed_h__

/**
 * This class uses the ancestor image to make 
 * truth information for triplets for
 * training a spatialembedding network
 * for clustering
 *
 */

#include <map>
#include <vector>

#include "larflow/PrepFlowMatchData/PrepMatchTriplets.h"

namespace larcv {
  class IOManager;
}
namespace larlite {
  class storage_manager;
}


namespace larflow {
  
namespace spatialembed {

  struct AncestorIDPix_t {
    int row;
    int col;
    int id;
    AncestorIDPix_t() :
    row(0),
      col(0),
      id(-1)
    {
    }
    AncestorIDPix_t( int r, int c, int i )
    : row(r),
      col(c),
      id(i)
    {};
    bool operator<( const AncestorIDPix_t& rhs ) {
      if (rhs.row<row) return true;
      else if ( rhs.row==row && rhs.col<col ) return true;
      else if ( rhs.row==row && rhs.col==col && rhs.id<id ) return true;
      return false;
    };
  };
  
  class PrepMatchEmbed {
  public:

    PrepMatchEmbed() {};
    virtual ~PrepMatchEmbed() {};

    void process( larcv::IOManager& iolcv,
                  larlite::storage_manager& ioll,
                  const PrepMatchTriplets& triplets );

    void _collect_ancestor_pixels( const larcv::Image2D& adc, const larcv::Image2D& ancestor );

    std::map<int,std::vector<AncestorIDPix_t> > pix_map_m[3]; 

    int get_num_instances(int plane)
    {
      if (plane>=0 && plane<3)
        return (int)pix_map_m[plane].size();
      return 0;
    };
    
    std::vector<int> get_id_list(int plane) {

      std::vector<int> id_v;
      if ( plane<0 || plane>=3 ) return id_v;
      id_v.reserve( pix_map_m[plane].size() );
      for ( auto it=pix_map_m[plane].begin(); it!=pix_map_m[plane].end(); it++ ) {
        id_v.push_back( it->first );
      }
      return id_v;
      
    };

    const std::vector<AncestorIDPix_t>& get_instance_pixlist(int plane, int aid );

    std::vector<int> _triplet_ancestor_id;
    std::map<int, std::vector<int> > _ancestor2tripletidx_m;
    void _assign_triplet_ancestor_id( const PrepMatchTriplets& tripletdata,
                                      const std::vector<larcv::Image2D>& ancestor_v );
                                     
    
  };
  
}
}
    

#endif
