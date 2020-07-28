#include <iostream>
#include <string>
#include <cmath>
#include <map>
#include <utility>
#include <iterator>
#include <algorithm>
#include "TFile.h"
#include "TH2D.h"


#include "larlite/core/DataFormat/storage_manager.h"
#include "larlite/core/DataFormat/larflow3dhit.h"
#include "larlite/core/DataFormat/mctrack.h"
#include "larlite/core/DataFormat/mcshower.h"


int main( int nargs, char** argv ) {

  typedef struct hit_t{
    int istruth;

    int tick;
    int U;
    int V;
    int Y;

    std::array<float,3> xyz;
    float score;

    hit_t()
      : istruth(0),
	tick(0),U(0),V(0),Y(0),
	score(0)
    {};
  } hit_t;
  

  std::cout << "larfow truth data" << std::endl;
  if ( nargs==1 ) {
    std::cout << "=== ARGUMENTS ===" << std::endl;
    std::cout <<  "truthana_larmatch [larmatch] [start entry] [num entries]" << std::endl;
    return 0;
  }

  std::string input_larmatch = argv[1];
  //std::string input_mcinfo   = argv[3];
  std::string adc_name = "wiremc";
  std::string chstatus_name = "wiremc";
  int startentry = atoi(argv[2]);
  int maxentries = atoi(argv[3]);

  
  larlite::storage_manager llio( larlite::storage_manager::kREAD );
  llio.add_in_filename( input_larmatch );
  //llio.add_in_filename( input_mcinfo );
  llio.open();

  int nentries = llio.get_entries();

  // output
  TFile* outfile = new TFile(Form("out_truthana_evt%d-%d.root",startentry,startentry+maxentries-1),"recreate");

  // DEFINE HISTOGRAMS
  const int nhists = 4;
  // strings for histo names
  std::string str1[4] = {"u","v","y","3d"};
  std::string str2[4] = {"dx","dy","dz","3d"};
  // score output versus flow distance
  TH2D* hprob_v_coldist[ nhists ] = {nullptr};
  for (int n=0; n<=3; n++ ) {
    char name[100];
    sprintf( name, "hprob_v_coldist_%s", str1[n].c_str() );
    hprob_v_coldist[n] = new TH2D( name,  ";distance from true target wire (cm); match probability", 3334, 0, 1000, 100, 0.0, 1.0 );
  }
  
  // error in flow (using max match)
  TH1D* herrflow[ nhists ] = { nullptr };
  for (int n=0; n<=3; n++ ) {
    char name[100];
    sprintf( name, "herrflow_%s", str2[n].c_str() );
    herrflow[n] = new TH1D(name, ";distance from true triplet (cm)", 3334, 0, 1000 );

  }
  
  if(startentry+maxentries > nentries) maxentries = nentries - startentry;
  for (int ientry=startentry; ientry<startentry+maxentries; ientry++ ) {

    std::cout << "===========================================" << std::endl;
    std::cout << "[ Entry " << ientry << " ]" << std::endl;

    llio.go_to(ientry);

    larlite::event_larflow3dhit* lfhit_v =
      (larlite::event_larflow3dhit*)llio.get_data(larlite::data::kLArFlow3DHit,"larmatch");
    

    std::map<int, hit_t> map1;
    std::multimap<int,int> mmap1;
    std::multimap<std::pair<int,int>,int> mmap2;
    typedef std::multimap<int,int>::iterator Iterator1;
    typedef std::multimap<std::pair<int,int>,int>::iterator Iterator2;

    // loop over hits

    for ( size_t ihit=0; ihit< lfhit_v->size(); ihit++ ) {
      const larlite::larflow3dhit& lfhit = lfhit_v->at(ihit);
      
      hit_t hit;
      hit.tick = lfhit.tick;
      hit.U = lfhit.targetwire[0];
      hit.V = lfhit.targetwire[1];
      hit.Y = lfhit.targetwire[2];
      hit.istruth = (int)lfhit.truthflag;
      hit.score = lfhit.track_score;
      for(int x=0; x<3; x ++) hit.xyz[x] = lfhit[x];

      map1.insert(std::make_pair(ihit,hit));
      mmap1.insert(std::make_pair(hit.istruth,ihit));
      mmap2.insert(std::make_pair(std::make_pair(hit.tick,hit.Y),ihit));

    }
    /*
    //debug
    std::cout << map1.size() <<" "<<mmap1.size() <<" "<< mmap2.size() << std::endl;
    std::map<int,hit_t>::iterator it = map1.begin();
    while(it != map1.end()){
      std::cout << it->first <<" "<< it->second.istruth <<" "<< it->second.tick <<" "<< it->second.Y <<" "<< it->second.score << std::endl;
      it++;
    }    
    Iterator1 it1 = mmap1.begin();
    while(it1 != mmap1.end()){
      std::cout << it1->first <<" "<< it1->second << std::endl;
      it1++;
    }
    Iterator2 it2 = mmap2.begin();
    while(it2 != mmap2.end()){
      std::cout << it2->first.first <<" "<< it2->first.second <<" "<< it2->second << std::endl;
      it2++;
    }
    */
    
    //grab true triplets : istruth==1
    std::pair<Iterator1,Iterator1> query1 = mmap1.equal_range(1);
    int nhits_wtrueflow = std::distance(query1.first, query1.second);
    for(Iterator1 it = query1.first; it!= query1.second; it++){
      std::pair<int,int> coord = std::make_pair(map1.at(it->second).tick, map1.at(it->second).Y);

      //std::cout << it->first <<" "<< it->second <<" "<< coord.first <<" "<< coord.second << std::endl; 
      float bestscore=0;
      int bestidx=0;
      std::pair<Iterator2,Iterator2> query2 = mmap2.equal_range(coord);
      for(Iterator2 it = query2.first; it!= query2.second; it++){
	if(map1.at(it->second).score > bestscore){
	  bestscore = map1.at(it->second).score;
	  bestidx = it->second;
	}

      }
      float dx = sqrt(pow(map1.at(bestidx).xyz[0] - map1.at(it->second).xyz[0],2));
      float dy = sqrt(pow(map1.at(bestidx).xyz[1] - map1.at(it->second).xyz[1],2));
      float dz = sqrt(pow(map1.at(bestidx).xyz[2] - map1.at(it->second).xyz[2],2));
      float dist = sqrt(dx*dx + dy*dy + dz*dz);

      float dU = map1.at(bestidx).U - map1.at(it->second).U;
      float dV = map1.at(bestidx).V - map1.at(it->second).V;

            
      herrflow[0]->Fill( dx );
      herrflow[1]->Fill( dy );
      herrflow[2]->Fill( dz );
      herrflow[3]->Fill( dist );
      hprob_v_coldist[0]->Fill(fabs(dU)*0.3, bestscore);
      hprob_v_coldist[1]->Fill(fabs(dV)*0.3, bestscore);
      hprob_v_coldist[3]->Fill(dist, bestscore);
	
    }// end of loop over points

    std::cout << "hits with true flow: " << nhits_wtrueflow << std::endl;
    std::cout << "total hits: " << lfhit_v->size() <<" "<< map1.size() << std::endl;

  }//event loop

  outfile->Write();
  outfile->Close();
  
  //io.finalize();
  llio.close();
  
  std::cout << "FIN" << std::endl;
  return 0;
};
