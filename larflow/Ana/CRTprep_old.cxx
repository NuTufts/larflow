// Script that prepares input to the network
// Want: row, col, height, charge in voxel, flashVector

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

  std::string str1[3] = {"U","V","Y"};
  std::string str2[3] = {"x","y","z"};
  std::string str3[3] = {"x","z","z"};
  std::string str4[3] = {"y","y","x"};
  std::string str5[4] = {"1cm","3cm","5cm","10cm"};  
  
  //Input for ttrees:
  int row;
  int col;
  int height;
  int chargeInVoxel;
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
  TFile* outfile = new TFile(Form("CRTprepped_%s",input_crtfile.c_str()),"recreate");

  TTree *tree = new TTree("tree","prepped tree");                                                                                                                   
  tree->Branch("row", &row, "row/I");
  tree->Branch("col", &col, "col/I");
  tree->Branch("height", &height, "height/I");
  tree->Branch("chargeInVoxel", &chargeInVoxel, "chargeInVoxel/I"); // WARNING maybe a float??
  
 
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
	
      }


    }
           
  }

  outfile->Write();
  outfile->Close();
  
  llio.close();

  return 0;
}
