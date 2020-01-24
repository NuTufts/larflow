#include <iostream>
#include <string>
#include <cmath>

#include "TFile.h"
#include "TH2D.h"

#include "larflow/PrepFlowMatchData/PrepFlowMatchData.hh"

#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/EventImage2D.h"
#include "larcv/core/DataFormat/EventSparseImage.h"
#include "larcv/core/ROOTUtil/ROOTUtils.h"

#include "ublarcvapp/UBImageMod/EmptyChannelAlgo.h"

#include "larlite/core/DataFormat/storage_manager.h"
#include "larlite/core/DataFormat/larflow3dhit.h"
#include "larlite/core/DataFormat/mctrack.h"
#include "larlite/core/DataFormat/mcshower.h"

void track_shower_labels_from_instanceimage( const std::vector<larcv::Image2D>& adc_v,
                                             const std::vector<larcv::Image2D>& instance_v,
                                             const larlite::event_mctrack& evmctrack,
                                             const larlite::event_mcshower& evmcshower,
                                             std::vector<larcv::Image2D>& truthlabels ) {

  // map of instance ID to index
  std::map<int,int> trackmap;
  std::map<int,int> showermap;

  int idx=0; 
  for ( auto const& mct : evmctrack ) {
    trackmap[mct.TrackID()] = idx;
    idx++;
  }

  idx=0;
  for ( auto const& mcs : evmcshower ) {
    showermap[mcs.TrackID()] = idx;
    idx++;
  }

  truthlabels.clear();
  int nlabels = 0;
  for ( auto const& img : instance_v ) {
    larcv::Image2D labels(img.meta());
    labels.paint(0);

    for (int row=0;row<(int)img.meta().rows(); row++) {
      for (int col=0; col<(int)img.meta().cols(); col++) {
        int id = (int)img.pixel(row,col);
        if ( id<=0 ) continue;
        if ( trackmap.find(id)!=trackmap.end() ) {
          labels.set_pixel(row,col,2);
          nlabels++;
        }
        else if ( showermap.find(id)!=showermap.end() ) {
          labels.set_pixel(row,col,1);
          nlabels++;
        }
      }
    }

    truthlabels.emplace_back(std::move(labels));
  }
  std::cout << "made " << nlabels << " across all planes" << std::endl;
}

int main( int nargs, char** argv ) {

  std::cout << "larfow truth data" << std::endl;

  std::string input_supera   = argv[1];
  std::string input_lcvtruth = argv[2];
  std::string input_larmatch = argv[3];
  std::string input_mcinfo   = argv[4];

  larcv::IOManager io( larcv::IOManager::kBOTH, "io", larcv::IOManager::kTickBackward );
  io.add_in_file( input_supera );
  io.add_in_file( input_lcvtruth );  
  io.set_out_file( "ana_temp.root" );
  io.reverse_all_products();  
  io.initialize();

  larlite::storage_manager llio( larlite::storage_manager::kREAD );
  llio.add_in_filename( input_larmatch );
  llio.add_in_filename( input_mcinfo );
  llio.open();

  int nentries = llio.get_entries();

  // create prepflowmatchdata for each source plane
  std::vector<larflow::PrepFlowMatchData> prepmatchdata_v;
  for (size_t p=0; p<3; p++ ) {
    char prepname[50];
    sprintf(prepname,"soureplane%d",(int)p);
    larflow::PrepFlowMatchData prepmatchdata(prepname);
    prepmatchdata.setADCproducer("wire");
    prepmatchdata.setChStatusProducer("wire");    
    prepmatchdata.setSourcePlaneIndex( (int)p );
    prepmatchdata.setLArFlowproducer("larflow");
    prepmatchdata.hasMCtruth(true);
    prepmatchdata.useAnaTree(false); // do not output match data
    prepmatchdata.initialize();
    prepmatchdata_v.emplace_back( std::move(prepmatchdata) );
  }

  // bad image algo
  ublarcvapp::EmptyChannelAlgo badchmaker;

  // constants
  int flowdir_sourceplane[6] = { 0, 0, 1, 1, 2, 2 };
  int flowdir_targetplane[6] = { 1, 2, 0, 2, 0, 1 };
  
  // output
  TFile* outfile = new TFile("out_truthana.root","recreate");

  // DEFINE HISTOGRAMS
  const int nhists = larflow::PrepFlowMatchData::kNumFlows+1;
  
  // score output versus flow distance
  TH2D* hprob_v_coldist[ nhists ] = {nullptr};
  for (int n=0; n<=larflow::PrepFlowMatchData::kNumFlows; n++ ) {
    char name[100];
    sprintf( name, "hprob_v_coldist_%s", larflow::PrepFlowMatchData::getFlowDirName( (larflow::PrepFlowMatchData::FlowDir_t)n ).c_str() );
    hprob_v_coldist[n] = new TH2D( name,  ";distance from true target wire (cm); match probability", 1000, 0, 1000, 100, 0.5, 1.0 );
  }

  // score output versus num matches
  TH2D* hprob_v_nmatches_good[ nhists ] = { nullptr };
  TH2D* hprob_v_nmatches_bad[ nhists ]  = { nullptr };  
  for (int n=0; n<=larflow::PrepFlowMatchData::kNumFlows; n++ ) {
    char goodname[100];
    sprintf( goodname, "hprob_v_nmatches_good_%s", larflow::PrepFlowMatchData::getFlowDirName( (larflow::PrepFlowMatchData::FlowDir_t)n ).c_str() );    
    hprob_v_nmatches_good[n] = new TH2D( goodname, "", 100, 0, 100, 100, 0.0, 1.0 );

    char badname[100];
    sprintf( badname, "hprob_v_nmatches_bad_%s", larflow::PrepFlowMatchData::getFlowDirName( (larflow::PrepFlowMatchData::FlowDir_t)n ).c_str() );    
    hprob_v_nmatches_bad[n] = new TH2D( badname, "", 100, 0, 100, 100, 0.0, 1.0 );
  }
    
  // error in flow (using max match)
  TH1D* herrflow[ nhists ] = { nullptr };
  TH1D* herrflow_shape[ nhists ][2] = { nullptr };  
  for (int n=0; n<=larflow::PrepFlowMatchData::kNumFlows; n++ ) {
    char name[100];
    sprintf( name, "herrflow_bestmatch_%s", larflow::PrepFlowMatchData::getFlowDirName( (larflow::PrepFlowMatchData::FlowDir_t)n ).c_str() );    
    herrflow[n] = new TH1D(name, "", 1000, 0, 1000 );

    char showername[100];
    sprintf( showername, "herrflow_bestmatch_shower_%s", larflow::PrepFlowMatchData::getFlowDirName( (larflow::PrepFlowMatchData::FlowDir_t)n ).c_str() );    
    herrflow_shape[n][0] = new TH1D( showername, "", 1000, 0, 1000 );

    char trackname[100];
    sprintf( trackname, "herrflow_bestmatch_track_%s", larflow::PrepFlowMatchData::getFlowDirName( (larflow::PrepFlowMatchData::FlowDir_t)n ).c_str() );    
    herrflow_shape[n][1] = new TH1D( trackname, "", 1000, 0, 1000 );
  }
  
  // number of matches distribution
  TH1D* hnmatches[ nhists ] = { nullptr };
  for (int n=0; n<=larflow::PrepFlowMatchData::kNumFlows; n++ ) {
    char name[100];
    sprintf( name, "hnmatches_%s", larflow::PrepFlowMatchData::getFlowDirName( (larflow::PrepFlowMatchData::FlowDir_t)n ).c_str() );    
    hnmatches[n] = new TH1D( name, "", 100, 0, 100000 );
  }

  // for debug, can plot into whole image space
  TH2D* hsrc = new TH2D("src","",3456,0,3456,1008,2400,2400+1008*6);

  for (int ientry=0; ientry<nentries; ientry++ ) {

    std::cout << "===========================================" << std::endl;
    std::cout << "[ Entry " << ientry << " ]" << std::endl;
    io.read_entry(ientry);

    larcv::EventImage2D* ev_adc      = (larcv::EventImage2D*)io.get_data(larcv::kProductImage2D,"wire");
    larcv::EventImage2D* ev_flow     = (larcv::EventImage2D*)io.get_data(larcv::kProductImage2D,"larflow");
    larcv::EventImage2D* ev_ancestor = (larcv::EventImage2D*)io.get_data(larcv::kProductImage2D,"ancestor");
    larcv::EventImage2D* ev_segment  = (larcv::EventImage2D*)io.get_data(larcv::kProductImage2D,"segment");
    larcv::EventImage2D* ev_instance = (larcv::EventImage2D*)io.get_data(larcv::kProductImage2D,"instance");
    larcv::EventChStatus* ev_badch   = (larcv::EventChStatus*)io.get_data(larcv::kProductChStatus, "wire" );

    std::cout << "num images: "            << ev_adc->Image2DArray().size()      << std::endl;
    std::cout << "num flow truth images: " << ev_flow->Image2DArray().size()     << std::endl;
    std::cout << "num ancestor images: "   << ev_ancestor->Image2DArray().size() << std::endl;

    auto const& adc_v = ev_adc->Image2DArray();

    // handle bad image making
    std::vector<larcv::Image2D> badch_v;
    if ( false) {
      badch_v = badchmaker.makeBadChImage( 4, 3, 2400, 6*1008, 3456, 6, 1, *ev_badch );
      std::cout << "Number of badcv images: " << badch_v.size() << std::endl;
      std::vector<larcv::Image2D> gapch_v = badchmaker.findMissingBadChs( adc_v, badch_v, 10.0, 100 );
      for (size_t p=0; p<badch_v.size(); p++ ) {
        for (size_t c=0; c<badch_v[p].meta().cols(); c++ ) {
          if ( gapch_v[p].pixel(0,c)>0 )
            badch_v[p].paint_col(c,255);
        }
        prepmatchdata_v[p].provideBadChannelImages( badch_v );
        //prepmatchdata_v[p].process( io );
        //const std::vector<larflow::FlowMatchMap>& matchmap_v = prepmatchdata_v[p].getMatchData();
        //std::cout << "source plane[" << p << "] flowmap: len=" << matchmap_v.size() << std::endl;
      }
    }

    larcv::EventSparseImage* ev_sparseimg = (larcv::EventSparseImage*)io.get_data(larcv::kProductSparseImage,"larflow");
    llio.go_to(ientry);

    larlite::event_larflow3dhit* lfhit_v =
      (larlite::event_larflow3dhit*)llio.get_data(larlite::data::kLArFlow3DHit,"larmatch");
    
    larlite::event_mctrack* evmctrack   = (larlite::event_mctrack*)llio.get_data(larlite::data::kMCTrack,"mcreco");
    larlite::event_mcshower* evmcshower = (larlite::event_mcshower*)llio.get_data(larlite::data::kMCShower,"mcreco");

    std::vector<larcv::Image2D> labeled_v;
    track_shower_labels_from_instanceimage( ev_adc->Image2DArray(), ev_instance->Image2DArray(),
                                            *evmctrack, *evmcshower,
                                            labeled_v );

    // // for debug
    // //char labelname[50];
    // //sprintf(labelname,"labels_entry%d",ientry);
    // //std::vector<TH2D> hlabeled = larcv::as_th2d_v( labeled_v, labelname );
    // //outfile->cd();
    // //for ( auto& h : hlabeled ) h.Write();

    // loop over hits
    int nhits_wtrueflow = 0;
    for ( size_t ihit=0; ihit<lfhit_v->size(); ihit++ ) {
      const larlite::larflow3dhit& lfhit = lfhit_v->at(ihit);

      // for each hit, fill the relevant histogram
      std::vector<float> score_v(6,0);
      std::vector<float> trueflow_v(6,0);
      std::vector<float> recoflow_v(6,0);
      std::vector<int>   has_trueflow(6,0);

      int row = adc_v.front().meta().row( lfhit.tick );

      int bestflowdir = -1;
      float bestscore = 0.;
      bool hastrueflow = false;
      for (int i=0; i<6; i++ ) {
        score_v[i] = lfhit[3+i];

        if ( score_v[i]==0.0 ) {
          continue; // no flow evaluated for this direction, triple
        }

        int srcplane = flowdir_sourceplane[i];
        int tarplane = flowdir_targetplane[i];
        
        // need to get true flow, from this direction
        auto const& flowimg = ev_flow->at(i);
        trueflow_v[i] = flowimg.pixel( row, lfhit.targetwire[ srcplane ] );
        if ( trueflow_v[i]!=0 ) {
          has_trueflow[i] = 1;
          hastrueflow = true;
        }

        // is it a good match in this direction?
        recoflow_v[i] = lfhit.targetwire[ tarplane ] - lfhit.targetwire[ srcplane ];

        if ( score_v[i]>bestscore ) {
          bestscore = score_v[i];
          bestflowdir = i;
        }
      }

      if ( hastrueflow )
        nhits_wtrueflow++;

      for ( int i=0; i<6; i++ ) {

        float flowerr = fabs(recoflow_v[i]-trueflow_v[i]);
        
        bool isgood = false;
        if ( flowerr<1.5 ) {
          isgood = true;
        }

        // start to fill histograms
        if ( has_trueflow[i]==1 ) {
          hprob_v_coldist[i]->Fill( flowerr, score_v[i] );
          hprob_v_coldist[6]->Fill( flowerr, score_v[i] );
        }

      }//end of score loop

      if ( has_trueflow[bestflowdir] ) 
        herrflow[6]->Fill( fabs(recoflow_v[bestflowdir]-trueflow_v[bestflowdir]) );
      
    }// end of loop over points

    std::cout << "hits with true flow: " << nhits_wtrueflow << std::endl;
    std::cout << "total hits: " << lfhit_v->size() << std::endl;

  }//event loop

  //hprob_v_coldist->Write();
  outfile->Write();
  outfile->Close();
  
  io.finalize();
  llio.close();
  
  std::cout << "FIN" << std::endl;
  return 0;
};
