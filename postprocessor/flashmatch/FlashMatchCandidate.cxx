#include "FlashMatchCandidate.h"

namespace larflow {

  FlashMatchCandidate::FlashMatchCandidate( const FlashData_t& fdata, const QCluster_t& qdata ) :
    _flashdata(&fdata),
    _cluster(&qdata)
  {
    // start the hypothesis build -- welcome to heuristics city! aka hell.
  }

}
