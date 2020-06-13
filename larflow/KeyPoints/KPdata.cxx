#include "KPdata.h"

#include <sstream>

namespace larflow {
namespace keypoints {

  bool kpdata_compare_x( const KPdata* lhs, const KPdata* rhs ) {
    if ( lhs->keypt[0] < rhs->keypt[0] )
      return true;
    return false;
  }

  bool kpdata_compare_y( const KPdata* lhs, const KPdata* rhs ) {
    if ( lhs->keypt[1] < rhs->keypt[1] )
      return true;
    return false;
  }

  bool kpdata_compare_z( const KPdata* lhs, const KPdata* rhs ) {
    if ( lhs->keypt[2] < rhs->keypt[2] )
      return true;
    return false;
  }

  std::string KPdata::str() const
  {
    std::stringstream ss;
    ss << "KPdata[type=" << crossingtype << " pid=" << pid
       << " vid=" << vid
       << " isshower=" << is_shower
       << " origin=" << origin << "] "
       << " kptype=" << kptype << " ";

    if ( imgcoord.size()>0 )
      ss << " imgstart=(" << imgcoord[0] << ","
         << imgcoord[1] << ","
         << imgcoord[2] << ","
         << imgcoord[3] << ") ";
    
    if ( keypt.size()>0 )
      ss << " keypt=(" << keypt[0] << "," << keypt[1] << "," << keypt[2] << ") ";
    
    return ss.str();
  }
  
}
}
