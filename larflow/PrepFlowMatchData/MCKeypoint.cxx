#include "MCKeypoint.h"

#include <sstream>

namespace larflow {
namespace prep {

  /**
   * @brief print info about keypoint info to standard out
   */    
  std::string MCKeypoint::str() const
  {
    std::stringstream ss;
    ss << "KPdata[type=" << kptype << " pid=" << pid
       << " tid=" << trackid
       << " isshower=" << is_shower
       << " origin=" << origin << "] ";

    if ( imgcoord.size()>0 )
      ss << " imgstart=(" << imgcoord[0] << ","
         << imgcoord[1] << ","
         << imgcoord[2] << ","
         << " tick=" << tick 
         << " row=" << row
         << ") " << std::endl;
    
    if ( keypt_true.size()>0 )
      ss << " keypt=(" << keypt_true[0] << "," << keypt_true[1] << "," << keypt_true[2] << ") " << std::endl;

    if ( keypt_appear.size()>0 )
      ss << " keypt (appear)=(" << keypt_appear[0] << "," << keypt_appear[1] << "," << keypt_appear[2] << ") " << std::endl;

    return ss.str();
  }

}
}