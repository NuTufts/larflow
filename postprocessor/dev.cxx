#include <iostream>
#include <string>

#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/EventImage2D.h"

int main( int nargs, char** argv ) {

  std::cout << "larflow post-processor dev" << std::endl;
  
  std::string input_larflow_file = "";
  std::string input_larlite_file = "";
  
  larcv::IOManager larflow_io( larcv::IOManager::kREAD );
  larflow_io.add_in_file( input_larflow_file );

  // input data

  // adc images

  // get cluster atomics

  // 
  
  
  return 0;

}
