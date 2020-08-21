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

  // Input for ttree
  int hitsPerVoxel;
  
  std::string input_crtfile = argv[1];
  int startentry = atoi(argv[2]);
  int maxentries = atoi(argv[3]);

  larlite::storage_manager llio( larlite::storage_manager::kREAD );
  llio.add_in_filename( input_crtfile );
  llio.open();

  int nentries = llio.get_entries();
  
  TFile* outfile = new TFile(Form("crt_%d-%d.root",startentry,startentry+maxentries-1),"recreate");
  TTree *tree = new TTree("tree","tree of hits per voxel");
  tree->Branch("hitsPerVoxel", &hitsPerVoxel, "hitsPerVoxel/I");
 
  // Define histograms
  const int nhists = 3;
  std::string str1[3] = {"U","V","Y"};
  std::string str2[3] = {"x","y","z"};
  std::string str3[3] = {"x","z","z"};
  std::string str4[3] = {"y","y","x"};
  
  // wire hists
  TH1D* hitcount_wire_hist[ nhists ] = {nullptr};
  for (int n = 0; n < 3; n++ ) {
    char name[100];
    sprintf( name, "hitcount_wire_hist_%s", str1[n].c_str() );
    hitcount_wire_hist[n] = new TH1D( name, "wire #", wireBins[n], 0, wireMax[n]);
  }

  TH2D* hitcount_wire_th2d[ nhists ] = {nullptr};
  for (int n = 0; n < 3; n++ ) {
    char name[100];
    sprintf( name, "hitcount_wire_th2d_%s", str1[n].c_str() );
    hitcount_wire_th2d[n] = new TH2D( name, "wire #; tick", wireBins[n]*10, 0, wireMax[n], 1008, 2400, 8448);
  }

  // xyz hists
  TH1D* hitcount_xyz_hist[ nhists ] = {nullptr};
  for (int n = 0; n < 3; n++ ) {
    char name[100];
    sprintf( name, "hitcount_xyz_hist_%s", str2[n].c_str() );
    hitcount_xyz_hist[n] = new TH1D( name, ";position", xyzBins[n], xyzMin[n], xyzMax[n]);
  }

  TH2D* hitcount_xyz_th2d[ nhists ] = {nullptr};
  for (int n = 0; n < 3; n++ ) {
    char name[100];
    sprintf( name, "hitcount_xyz_th2d_%s%s", str3[n].c_str(), str4[n].c_str() );
    hitcount_xyz_th2d[n] = new TH2D( name, ";position ; position", xzzBins[n], xzzMin[n], xzzMax[n], yyxBins[n], yyxMin[n], yyxMax[n]);
  }

  TH3D* hitcount_xyz_th3d = nullptr;
  char name[100];
  sprintf( name, "hitcount_xyz_th3d");
  hitcount_xyz_th3d = new TH3D( name, ";position ;position ; position", (256/5), 0., 256., (234/5), -117., 117., (1036/5), 0., 1036.);

  // Loop over events
  for (int i = startentry; i < (startentry + maxentries); i++) {
    
    std::cout << "===========================================" << std::endl;
    std::cout << "[ Entry " << i << " ]" << std::endl;

    llio.go_to(i);

    larlite::event_larflowcluster* clusters_v = (larlite::event_larflowcluster*)llio.get_data(larlite::data::kLArFlowCluster,"fitcrttrack_larmatchhits");

    // loop thru clusters
    for ( size_t iCluster = 0; iCluster < clusters_v->size(); iCluster++ ) {

      const larlite::larflowcluster& cluster = clusters_v->at( iCluster );

      std::cout << "I'm in cluster: " << iCluster << std::endl;
      
      //      larlite::event_larflow3dhit* lfhits_v = (larlite::event_larflow3dhit*)llio.get_data(larlite::data::kLArFlow3DHit,"larmatch");

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

	// fill wire hists
	hitcount_wire_hist[0]->Fill(hit_U);
	hitcount_wire_hist[1]->Fill(hit_V);
	hitcount_wire_hist[2]->Fill(hit_Y);

	hitcount_wire_th2d[0]->Fill(hit_U, tick);
	hitcount_wire_th2d[1]->Fill(hit_V, tick);
	hitcount_wire_th2d[2]->Fill(hit_Y, tick);

	// fill xyz hists
	hitcount_xyz_hist[0]->Fill(hit_x);
	hitcount_xyz_hist[1]->Fill(hit_y);
	hitcount_xyz_hist[2]->Fill(hit_z);

	hitcount_xyz_th2d[0]->Fill(hit_x, hit_y);
	hitcount_xyz_th2d[1]->Fill(hit_z, hit_y);
	hitcount_xyz_th2d[2]->Fill(hit_z, hit_x);

	hitcount_xyz_th3d->Fill(hit_x, hit_y, hit_z);

      }


    }
           
  }

  // Outside event loop
  for (int i = 1; i <= (xyzBins[0]/5); i++) { // here use i = 1, i <= max, NOT i = 0, i < max (bc bin 0 is underflow in ROOT histograms)
    for (int j = 1; j <= (xyzBins[1]/5); j++) {
      for (int k = 1; k <= (xyzBins[2]/5); k++) {
	
	//	std::cout << hitcount_xyz_th3d->GetBinContent(i, j, k) << std::endl;
	hitsPerVoxel = hitcount_xyz_th3d->GetBinContent(i, j, k);
	tree->Fill();

      }
    }
  }
  
  outfile->Write();
  outfile->Close();
  
  llio.close();

  return 0;
}
