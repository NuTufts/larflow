#include <iostream>
#include <vector>
#include <set>

#include "larflow/Reco/KPSRecoManager.h"

int main( int nargs, char** argv ) {

  std::cout << "KPSRecoManager test" << std::endl;

  // we want to know
  // 1) which shower cluster, if any, matches best to the truth shower-trunk
  // 2) purity of that best shower cluster
  // 3) efficiency of that shower cluster
  // 4) pca-line versus shower direction
  // 5) statistics of matched clusters and false clusters
  larflow::reco::KPSRecoManager recoman( "out_kpsana.root", 2 );

  std::cout << "[enter] to finish" << std::endl;
  std::cin.get();
  
  return 0;
  
}
