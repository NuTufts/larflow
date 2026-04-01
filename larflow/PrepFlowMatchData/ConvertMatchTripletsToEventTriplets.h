#ifndef __LARFLOW_PREP_CONVERT_MATCH_TRIPLETS_TO_EVENT_TRIPLETS_H__
#define __LARFLOW_PREP_CONVERT_MATCH_TRIPLETS_TO_EVENT_TRIPLETS_H__

/**
* @class ConvertMatchTripletsToEventTriplets
* @ingroup prep
*
* @brief Convert data stored in PrepMatchTriplets to EventTriplets_t
*
* This class was written in order to allow old simulation files from 
* the official MicroBooNE production to be used in the new training data
* preparation pipeline made for the new simulated data production done at Tufts.
*
* The old production saved 3D truth labels for the 3D ionization deposits by
* projecting the labels into the pixels of the 2D wire plane images. This
* was done to save in part to save disk space but was also due to the fact that
* the focus of ML development at the time was on parsing of 2D images.
* The new production now saves the 3D energy deposition information, making label making
* much easier.
*
* But the old production files are necessary to push results from new developments,
* namely the shower origin model, into the analysis workflow for MicroBooNE.
*
* This class is meant to open a file with larflow::prep::PrepMatchTriplets class
* made using test/run_prepmatchtriplets_wfulltruth.py. 
* 
*
*/

#include "larcv/core/Base/larcv_base.h"

#include "ublarcvapp/MCTools/EventMCPixelLabels.h"
#include "ublarcvapp/MCTools/MCPixelLabelMaker.h"


namespace larflow {
namespace prep {

class PrepMatchTriplets; // forward declaration

class ConvertMatchTripletsToEventTriplets : public larcv::larcv_base {

public:
  ConvertMatchTripletsToEventTriplets()
  : larcv::larcv_base("ConvertMatchTripletsToEventTriplets")
  {}

  ~ConvertMatchTripletsToEventTriplets(){};

  static void convert( 
    ublarcvapp::mctools::MCPixelLabelMaker & mclabelmaker, 
    larflow::prep::PrepMatchTriplets& tripletmaker );


};

}    
}

#endif