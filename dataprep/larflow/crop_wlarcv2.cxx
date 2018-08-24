#include <iostream>
#include <string>
#include <sys/stat.h>

// larcv
#include "larcv/core/DataFormat/EventImage2D.h"
#include "larcv/core/DataFormat/Image2D.h"
#include "larcv/core/DataFormat/BBox.h"
#include "larcv/core/Processor/ProcessDriver.h"

// llcv
#include "Base/DataCoordinator.h"

int main( int nargs, char** argv ) {

  std::cout << "[ Select Tune Sample ]" << std::endl;

  std::string SUP_FILE   = argv[1];  // larflow source images
  std::string CFG_FILE   = argv[2];  // config file
  std::string OUT_FILE   = argv[3];  // output cropped file

  // make sure out_file doesn't already exist
  struct stat buffer;   
  bool outputexists = ( stat (OUT_FILE.c_str(), &buffer) == 0);
  if ( outputexists ) {
    std::cout << "Output file already exists. Preventing overwriting by stopping program." << std::endl;
    return 1;
  }

  std::vector<std::string> input_files;
  input_files.push_back( SUP_FILE );
  
  larcv::ProcessDriver driver("LArFlowCrop");
  driver.configure( CFG_FILE );
  driver.override_input_file( input_files );

  driver.override_output_file( OUT_FILE );
  driver.initialize();

  bool ok = driver.process_entry();
  while (ok) {
    ok = driver.process_entry();
  }

  driver.finalize();
  
  return 0;
}
