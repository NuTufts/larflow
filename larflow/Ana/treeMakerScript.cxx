// Code to create the tree of hitsPerVoxel from an input TH3D                                                                                                   

#include <iostream>
#include <string>
#include <cmath>
#include <vector>

#include "TFile.h"
#include "TTree.h"
#include "TH3D.h"

#include "larlite/core/DataFormat/storage_manager.h"

int main( int nargs, char** argv ) {

  const int voxHists = 4;
  std::string str5[4] = {"1cm","3cm","5cm","10cm"};
  int voxelSize[] = {1, 3, 5, 10}; // cm

  float xyzMin[] = {0., -117., 0.};
  float xyzMax[] = {256., 117., 1036.};
  int xyzBins[] = {256, 234, 1036};

  // Input for ttrees
  int hitsPerVoxel[4];

  std::string input_th3d_file = argv[1];

  TFile *inputFile = new TFile(input_th3d_file.c_str(),"READ");

  //larlite::storage_manager llio( larlite::storage_manager::kREAD );
  //llio.add_in_filename( input_th3d_file );
  //llio.open();

  //  int nentries = llio.get_entries();
  // std::cout << "This is nentries: " << nentries << std::endl;

  // Remake variables in the input file
  TH3D* hitcount_xyz_th3d[ voxHists ] = {nullptr};
  for (int n = 0; n < voxHists; n++ ) {
    char name[100];
    sprintf( name, "hitcount_xyz_th3d_%s", str5[n].c_str() );
    hitcount_xyz_th3d[n] = new TH3D( name, ";position ;position ; position", (xyzBins[0]/voxelSize[n]), xyzMin[0], xyzMax[0], (xyzBins[1]/voxelSize[n]), xyzMin[1], xyzMax[1], (xyzBins[2]/voxelSize[n]), xyzMin[2], xyzMax[2]);
  }
  
  TFile* outfile = new TFile(Form("tree_%s",input_th3d_file.c_str()),"recreate");

  TTree* T[ voxHists ] = {nullptr};
  for ( int i = 0; i < voxHists; i++ ) {
    T[i] = new TTree(Form("T_%s",str5[i].c_str()),"");
    T[i]->Branch("hitsPerVoxel",&hitsPerVoxel[i],"hitsPerVoxel/I");
  }

  for (int n = 0; n < voxHists; n++) {

    for (int i = 1; i <= (xyzBins[0]/voxelSize[n]); i++) { // here use i = 1, i <= max, NOT i = 0, i < max (bc bin 0 is underflow in ROOT histograms)
      for (int j = 1; j <= (xyzBins[1]/voxelSize[n]); j++) {
        for (int k = 1; k <= (xyzBins[2]/voxelSize[n]); k++) {

          hitsPerVoxel[n] = hitcount_xyz_th3d[n]->GetBinContent(i, j, k);
          T[n]->Fill();

        }
      }
    }

  }

  outfile->Write();
  outfile->Close();

  return 0;
}
