// Want to histogram hits in 3D: # hits as a function of position for the 3 planes 

#include <iostream>
#include <string>
#include <cmath>
#include <vector>

#include "TFile.h"
#include "TTree.h"

#include "TH1D.h"
#include "TH2D.h"
#include "TH3D.h"

#include "larlite/core/DataFormat/storage_manager.h"
#include "larlite/core/DataFormat/larflow3dhit.h"
#include "larlite/core/DataFormat/larflowcluster.h"

int main( int nargs, char** argv ) {

  int hit_U = 0;
  int hit_V = 0;
  int hit_Y = 0;
  int tick = 0;

  float hit_x = 0.;
  float hit_y = 0.;
  float hit_z = 0.;

  // Define variables for histograms
  int wireBins[] = {250, 250, 350};
  int wireMax[] = {2500, 2500, 3500};

  float xyzMin[] = {0., -117., 0.};
  float xyzMax[] = {256., 117., 1036.};
  int xyzBins[] = {256, 234, 1036};
  
  float xzzMin[] = {0., 0., 0.}; 
  float xzzMax[] = {256., 1036., 1036.};
  int xzzBins[] = {256, 1036, 1036};
  
  float yyxMin[] = {-117., -117., 0.}; 
  float yyxMax[] = {117., 117., 256.};
  int yyxBins[] = {234, 234, 256};

  int voxelSize[] = {1, 3, 5, 10}; // cm

  const int nhists = 3;
  const int voxHists = 4;
  
  std::string str1[3] = {"U","V","Y"};
  std::string str2[3] = {"x","y","z"};
  std::string str3[3] = {"x","z","z"};
  std::string str4[3] = {"y","y","x"};
  std::string str5[4] = {"1cm","3cm","5cm","10cm"};  
  
  // Input for ttrees
  //  int hitsPerVoxel[4];
  //  int hitsPer1cmVoxel; // DEBUG
  
  std::string crtfile_path = argv[1];
  std::string input_crtfile = argv[2];
  //  int startentry = atoi(argv[2]);
  //int maxentries = atoi(argv[3]);
  //  std::string output_filename = argv[2]

  larlite::storage_manager llio( larlite::storage_manager::kREAD );
  llio.add_in_filename( crtfile_path + input_crtfile );
  llio.open();

  int nentries = llio.get_entries();
  //  std::cout << "[DEBUG] This is nentries: " << nentries << std::endl;
  
  //  TFile* outfile = new TFile(Form("crt_%d-%d.root",startentry,startentry+maxentries-1),"recreate");
  TFile* outfile = new TFile(Form("CRTana_%s",input_crtfile.c_str()),"recreate");

  /*
  TTree* T[ voxHists ] = {nullptr};
  for ( int i = 0; i < voxHists; i++ ) {
    T[i] = new TTree(Form("T_%s",str5[i].c_str()),"");
    T[i]->Branch("hitsPerVoxel",&hitsPerVoxel[i],"hitsPerVoxel/I");
  }
  */

  // [DEBUG] singular tree
  //  TTree *tree = new TTree("tree","tree of hits per voxel");
  //tree->Branch("hitsPer1cmVoxel", &hitsPer1cmVoxel, "hitsPer1cmVoxel/I");
  
  // wire hists
  TH1D* hitcount_wire_hist[ nhists ] = {nullptr};
  for (int n = 0; n < nhists; n++ ) {
    char name[100];
    sprintf( name, "hitcount_wire_hist_%s", str1[n].c_str() );
    hitcount_wire_hist[n] = new TH1D( name, "wire #", wireBins[n], 0, wireMax[n]);
  }

  TH2D* hitcount_wire_th2d[ nhists ] = {nullptr};
  for (int n = 0; n < nhists; n++ ) {
    char name[100];
    sprintf( name, "hitcount_wire_th2d_%s", str1[n].c_str() );
    hitcount_wire_th2d[n] = new TH2D( name, "wire #; tick", wireBins[n]*10, 0, wireMax[n], 1008, 2400, 8448);
  }

  // xyz hists
  TH1D* hitcount_xyz_hist[ nhists ] = {nullptr};
  for (int n = 0; n < nhists; n++ ) {
    char name[100];
    sprintf( name, "hitcount_xyz_hist_%s", str2[n].c_str() );
    hitcount_xyz_hist[n] = new TH1D( name, ";position", xyzBins[n], xyzMin[n], xyzMax[n]);
  }

  TH2D* hitcount_xyz_th2d[ nhists ] = {nullptr};
  for (int n = 0; n < nhists; n++ ) {
    char name[100];
    sprintf( name, "hitcount_xyz_th2d_%s%s", str3[n].c_str(), str4[n].c_str() );
    hitcount_xyz_th2d[n] = new TH2D( name, ";position ; position", xzzBins[n], xzzMin[n], xzzMax[n], yyxBins[n], yyxMin[n], yyxMax[n]);
  }

  TH3D* hitcount_xyz_th3d[ voxHists ] = {nullptr};
  for (int n = 0; n < voxHists; n++ ) {
    char name[100];
    sprintf( name, "hitcount_xyz_th3d_%s", str5[n].c_str() );
    hitcount_xyz_th3d[n] = new TH3D( name, ";position ;position ; position", (xyzBins[0]/voxelSize[n]), xyzMin[0], xyzMax[0], (xyzBins[1]/voxelSize[n]), xyzMin[1], xyzMax[1], (xyzBins[2]/voxelSize[n]), xyzMin[2], xyzMax[2]);
  }

  // Loop over events
  //  for (int i = startentry; i < (startentry + maxentries); i++) {
  for (int i = 0; i < nentries; i++) {
    
    //    std::cout << "===========================================" << std::endl;
    //std::cout << "[ Entry " << i << " ]" << std::endl;

    llio.go_to(i);

    larlite::event_larflowcluster* clusters_v = (larlite::event_larflowcluster*)llio.get_data(larlite::data::kLArFlowCluster,"fitcrttrack_larmatchhits");

    // loop thru clusters
    for ( size_t iCluster = 0; iCluster < clusters_v->size(); iCluster++ ) {

      const larlite::larflowcluster& cluster = clusters_v->at( iCluster );

      // std::cout << "I'm in cluster: " << iCluster << std::endl;
      
      // loop thru hits in this cluster
      for ( size_t iHit = 0; iHit < cluster.size(); iHit++ ) {

	const larlite::larflow3dhit& lfhit = cluster.at( iHit );

	hit_U = lfhit.targetwire[0];
	hit_V = lfhit.targetwire[1];
	hit_Y = lfhit.targetwire[2];
	tick = lfhit.tick;
	hit_x = lfhit[0];
	hit_y = lfhit[1];
	hit_z = lfhit[2];
	
	// fill wire 1d hists
	hitcount_wire_hist[0]->Fill(hit_U);
	hitcount_wire_hist[1]->Fill(hit_V);
	hitcount_wire_hist[2]->Fill(hit_Y);

	// fill wire 2d hists
	hitcount_wire_th2d[0]->Fill(hit_U, tick);
	hitcount_wire_th2d[1]->Fill(hit_V, tick);
	hitcount_wire_th2d[2]->Fill(hit_Y, tick);

	// fill xyz 1d hists
	hitcount_xyz_hist[0]->Fill(hit_x);
	hitcount_xyz_hist[1]->Fill(hit_y);
	hitcount_xyz_hist[2]->Fill(hit_z);

	// fill xyz 2d hists
	hitcount_xyz_th2d[0]->Fill(hit_x, hit_y);
	hitcount_xyz_th2d[1]->Fill(hit_z, hit_y);
	hitcount_xyz_th2d[2]->Fill(hit_z, hit_x);
	
	// fill 3d hists for 4 diff voxel sizes
	hitcount_xyz_th3d[0]->Fill(hit_x, hit_y, hit_z);
	hitcount_xyz_th3d[1]->Fill(hit_x, hit_y, hit_z);
	hitcount_xyz_th3d[2]->Fill(hit_x, hit_y, hit_z);
	hitcount_xyz_th3d[3]->Fill(hit_x, hit_y, hit_z);
	
      }


    }
           
  }

  // Outside event loop
  // This is for creating a tree AFTER all input files have been hadded! (doesn't work otherwise)

  /*
  
  for (int n = 0; n < voxHists; n++) {
    
    for (int i = 1; i <= (xyzBins[0]/voxelSize[0]); i++) { // here use i = 1, i <= max, NOT i = 0, i < max (bc bin 0 is underflow in ROOT histograms)
      for (int j = 1; j <= (xyzBins[1]/voxelSize[0]); j++) {
	for (int k = 1; k <= (xyzBins[2]/voxelSize[0]); k++) {

	  hitsPerVoxel[n] = hitcount_xyz_th3d[n]->GetBinContent(i, j, k);
	  T[n]->Fill();

	  hitsPer1cmVoxel = hitcount_xyz_th3d[0]->GetBinContent(i, j, k);
	  tree->Fill();
	  
	}
      }
    }
    
  }

  */
  
  outfile->Write();
  outfile->Close();
  
  llio.close();

  return 0;
}
