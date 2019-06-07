#include "ContourFlowMatch.h"

namespace larflow {

  /**
   * constructor
   *
   * contour indices refers to the contour vectors in
   * ublarcvapp/ContourTools/ContourClusterAlgo::m_plane_contour_v
   *
   * @param[in] srcid Index of source image contour
   * @param[in] tarid Index of target image contour
   *
   */
  ContourFlowMatch_t::ContourFlowMatch_t( int srcid, int tarid )
    : src_ctr_id(srcid), tar_ctr_id(tarid), score(0)
  {}

  /**
   * copy constructor
   *
   */
  ContourFlowMatch_t::ContourFlowMatch_t( const ContourFlowMatch_t& x )
    : src_ctr_id(x.src_ctr_id),
      tar_ctr_id(x.tar_ctr_id),
      score(x.score)
  {
    matchingflow_v = x.matchingflow_v;
  }
  

}
