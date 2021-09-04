#ifndef __CRVARSMAKER_H__
#define __CRVARSMAKER_H__

#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include <math.h>
// ROOT

#include "TH2F.h"
#include "TCanvas.h"
#include "TStyle.h"

// larutil
#include "larlite/LArUtil/LArProperties.h"
#include "larlite/LArUtil/DetectorProperties.h"
#include "larlite/LArUtil/Geometry.h"
#include "larlite/LArUtil/ClockConstants.h"
#include "larlite/LArUtil/SpaceChargeMicroBooNE.h"

// larcv
#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/EventClusterMask.h"
#include "larcv/core/DataFormat/EventSparseImage.h"
#include "larcv/core/DataFormat/EventImage2D.h"
#include "larcv/core/DataFormat/EventPGraph.h"
#include "larcv/core/Base/larcv_logger.h"
// larlite
#include "larlite/DataFormat/storage_manager.h"

// ublarcvapp
#include "ublarcvapp/MCTools/NeutrinoVertex.h"
#include "ublarcvapp/dbscan/sDBScan.h"
// Misc
#include "CRSuiteFunctions.h"
namespace larflow {
namespace cosmicremovalsuite {


class CRVarsMaker {
public:
  CRVarsMaker() {};
  ~CRVarsMaker() {}


  int run_varsmaker(int mode=0, int file_limit=1, int start_file=0);
};

void print_signal();

}
}//End namespaces
#endif
