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
#include "LArUtil/LArProperties.h"
#include "LArUtil/DetectorProperties.h"
#include "LArUtil/Geometry.h"
#include "LArUtil/ClockConstants.h"
#include "LArUtil/SpaceChargeMicroBooNE.h"

// larcv
#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/EventClusterMask.h"
#include "larcv/core/DataFormat/EventSparseImage.h"
#include "larcv/core/DataFormat/EventImage2D.h"
#include "larcv/core/DataFormat/EventPGraph.h"
#include "larcv/core/Base/larcv_logger.h"
// larlite
#include "DataFormat/storage_manager.h"
#include "DataFormat/mctruth.h"

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


  int run_varsmaker_rootfile(bool IsMC,
          std::string infile_dlreco,
  				std::string infile_mrcnn,
  				std::string OutFileName,
  				std::string OutTreeName
  			);
  std::vector<double> run_varsmaker_arrsout(bool IsMC,
          larcv::IOManager* io_mrcnn,
          larcv::IOManager* io_dlreco,
          larlite::storage_manager& ioll_dlreco,
          int entry_num=-1
  			);
  Cosmic_Products cosmic_products_getter;

  // larcv::IOManager* io_mrcnn  ;//= new larcv::IOManager(larcv::IOManager::kREAD,"IOManager_MRCNN", larcv::IOManager::kTickBackward);
  // larcv::IOManager* io_dlreco ;// = new larcv::IOManager(larcv::IOManager::kREAD,"IOManager_DLRECO", larcv::IOManager::kTickBackward);
  // larlite::storage_manager ioll_dlreco ;// = larlite::storage_manager(larlite::storage_manager::kREAD);

};

void print_signal();

}
}//End namespaces
#endif
