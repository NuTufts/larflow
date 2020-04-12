#include "KPdata.h"

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
  
  
}
}
