#include <iostream>

// larlite
#include "LArUtil/Geometry.h"

// ROOT
#include "TFile.h"
#include "TTree.h"

int main(int nargs, char** argv ) {

  const larutil::Geometry* geo = larutil::Geometry::GetME();

  TFile out("consistency3d_data.root","recreate");

  TTree t("consist3d","3D consistency data");

  std::vector<int> nwires_v = { 2400, 2400, 3456 };

  float posyz[2];
  int source_plane;
  int target_plane;

  t.Branch("source_plane",&source_plane,"source_plane/I");
  t.Branch("target_plane",&target_plane,"target_plane/I");
  t.Branch("posyz", posyz, "posyz[2]/F");

  double ypos;
  double zpos;
  for (int y=0; y<3456; y++) {
    for (int puv=0; puv<2; puv++) {
      target_plane = puv;
      for (int wire=0; wire<nwires_v[puv]; wire++) {
	geo->IntersectionPoint( y, wire, 2, puv, ypos, zpos );
	posyz[0] = ypos;
	posyz[1] = zpos;
	t.Fill();
      }
    }
  }
  
  t.Write();

  return 0;
}
