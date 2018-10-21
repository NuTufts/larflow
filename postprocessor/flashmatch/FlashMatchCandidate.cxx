#include "FlashMatchCandidate.h"

// larlite
#include "LArUtil/Geometry.h"
#include "LArUtil/LArProperties.h"
#include "LArUtil/SpaceChargeMicroBooNE.h"
#include "LArUtil/TimeService.h"
#include "SelectionTool/OpT0Finder/PhotonLibrary/PhotonVisibilityService.h"


namespace larflow {



  FlashMatchCandidate::FlashMatchCandidate( const FlashData_t& fdata, const QClusterCore& qcoredata ) :
    _flashdata(&fdata),
    _cluster(qcoredata._cluster),
    _core(&qcoredata)
  {
    // when dealing with information from the core, we will need this offset
    _xoffset = (_flashdata->tpc_tick-3200)*0.5*larutil::LArProperties::GetME()->DriftVelocity();
  }

  

  
}
