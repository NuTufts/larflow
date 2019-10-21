#include <iostream>
#include <string>
#include <cmath>

#include "TFile.h"
#include "TH2D.h"

#include "larflow/PrepFlowMatchData/PrepFlowMatchData.hh"

#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/EventImage2D.h"
#include "larcv/core/DataFormat/EventSparseImage.h"

#include "larlite/core/DataFormat/storage_manager.h"
#include "larlite/core/DataFormat/larflow3dhit.h"

int main( int nargs, char** argv ) {

  std::cout << "larfow truth data" << std::endl;

  std::string input_supera   = argv[1];
  std::string input_lcvtruth = argv[2];
  std::string input_larmatch = argv[3];

  larcv::IOManager io( larcv::IOManager::kBOTH, "io", larcv::IOManager::kTickBackward );
  io.add_in_file( input_supera );
  io.add_in_file( input_lcvtruth );  
  io.set_out_file( "ana_temp.root" );
  io.reverse_all_products();  
  io.initialize();

  larlite::storage_manager llio( larlite::storage_manager::kREAD );
  llio.add_in_filename( input_larmatch );
  llio.open();

  int nentries = llio.get_entries();

  larflow::PrepFlowMatchData prepmatchdata("larmatch");
  prepmatchdata.setADCproducer("wire");
  prepmatchdata.setLArFlowproducer("larflow");
  prepmatchdata.hasMCtruth(true);
  prepmatchdata.useAnaTree(false);
  prepmatchdata.initialize();

  // output
  TFile* outfile = new TFile("out_truthana.root","recreate");

  // score output versus flow distance
  TH2D* hprob_v_coldist  = new TH2D( "hprob_v_coldist",  "", 1000, 0, 1000, 100, 0.5, 1.0 );

  // score output versus num matches
  TH2D* hprob_v_nmatches_good[2] = { new TH2D( "hprob_v_nmatches_good_y2u", "", 100, 0, 100, 100, 0.5, 1.0 ),
                                     new TH2D( "hprob_v_nmatches_good_y2v", "", 100, 0, 100, 100, 0.5, 1.0 ) };
  TH2D* hprob_v_nmatches_bad[2]  = { new TH2D( "hprob_v_nmatches_bad_y2u", "", 100, 0, 100, 100, 0.5, 1.0 ),
                                     new TH2D( "hprob_v_nmatches_bad_y2v", "", 100, 0, 100, 100, 0.5, 1.0 ) };

  // error in flow (using max match)
  TH1D* herrflow[2] = { new TH1D( "herrflow_y2u", "", 1000, 0, 1000 ),
                        new TH1D( "herrflow_y2v", "", 1000, 0, 1000 ) };
  
  // error in flow (using max match), split by track and shower true pixels
  TH1D* herrflow_shape[2][2] = { { new TH1D( "herrflow_y2u_track", "", 1000, 0, 1000 ), new TH1D( "herrflow_y2u_shower", "", 1000, 0, 1000 ) },
                                 { new TH1D( "herrflow_y2v_track", "", 1000, 0, 1000 ), new TH1D( "herrflow_y2v_shower", "", 1000, 0, 1000 ) } };
  
  // number of matches distribution
  TH1D* hnmatches[2] = { new TH1D( "hnmatches_y2u", "", 100, 0, 100 ),
                         new TH1D( "hnmatches_y2v", "", 100, 0, 100 ) };

  // for debug, can plot into whole image space
  TH2D* hsrc = new TH2D("src","",3456,0,3456,1008,2400,2400+1008*6);
  
  for (int ientry=0; ientry<nentries; ientry++ ) {

    std::cout << "===========================================" << std::endl;
    std::cout << "[ Entry " << ientry << " ]" << std::endl;
    io.read_entry(ientry);

    larcv::EventImage2D* ev_adc      = (larcv::EventImage2D*)io.get_data(larcv::kProductImage2D,"wire");
    larcv::EventImage2D* ev_flow     = (larcv::EventImage2D*)io.get_data(larcv::kProductImage2D,"larflow");
    larcv::EventImage2D* ev_ancestor = (larcv::EventImage2D*)io.get_data(larcv::kProductImage2D,"ancestor");

    std::cout << "num images: "            << ev_adc->Image2DArray().size()      << std::endl;
    std::cout << "num flow truth images: " << ev_flow->Image2DArray().size()     << std::endl;
    std::cout << "num ancestor images: "   << ev_ancestor->Image2DArray().size() << std::endl;

    prepmatchdata.process( io );
    const std::vector<larflow::FlowMatchMap>& matchmap_v = prepmatchdata.getMatchData();
    std::cout << "flowmap: len=" << matchmap_v.size() << std::endl;

    larcv::EventSparseImage* ev_sparseimg = (larcv::EventSparseImage*)io.get_data(larcv::kProductSparseImage,"larflow");

    llio.go_to(ientry);

    larlite::event_larflow3dhit* lfhit_v[2] =
      { (larlite::event_larflow3dhit*)llio.get_data(larlite::data::kLArFlow3DHit,"larmatchy2u"),
        (larlite::event_larflow3dhit*)llio.get_data(larlite::data::kLArFlow3DHit,"larmatchy2v") };

    const larcv::SparseImage& sparse_src = ev_sparseimg->at(0);

    for (int iflow=0; iflow<2; iflow++ ) {

      // loop over source indices and info in matchdata      
      // calculate:
      // -- for each source pixel with flow, number of ancestor particles to match against
      //    versus prob score for false and true examples. (towards hard-example mining/weighting)

      
      // loop over hits
      // calculate:
      //  -- prob score versus distance between match column and true flow column
      larlite::event_larflow3dhit* lfv = lfhit_v[iflow];
      std::cout << "larflowhits[iflow=" << iflow << "] len=" << lfv->size() << std::endl;

      std::map< std::array<int,2>, float > source_max_score;    // key=(col,row) value=score
      std::map< std::array<int,2>, int >   source_correct;      // key=(col,row) value={1=correct}
      std::map< std::array<int,2>, float > source_max_errflow;  // key=(col,row) value=trueflow for max match
      
      for (int ihit=0; ihit<(int)lfv->size(); ihit++ ) {
        const larlite::larflow3dhit& hit = lfv->at(ihit);

        float tick = (float)hit.tick;
        int srccol = hit.srcwire;
        int tarcol = hit.targetwire[iflow];

        int row = (int)ev_adc->Image2DArray().at(2).meta().row( tick );
        if ( row>=ev_adc->Image2DArray().at(2).meta().rows() ) {
          //std::cout << "max row stored" << std::endl;
          continue;
        }

        std::array<int,2> srccoord = { srccol, row };

        int trueflow = ev_flow->Image2DArray().at(4+iflow).pixel(row,srccol);
        
        int truecol  = srccol + trueflow;

        float dcol = std::fabs( tarcol-truecol );
        float prob = hit.track_score;

        if ( source_max_score.find( srccoord )==source_max_score.end() ) {
          // first entry
          source_max_score[ srccoord ] = prob;
          source_correct[ srccoord ]   = ( dcol<5 ) ? 1 : 0;
          source_max_errflow[ srccoord ] = fabs( tarcol-truecol );  // error in predicted flow
        }
        else {
          if ( source_max_score[ srccoord ]<prob ) {
            source_max_score[ srccoord ] = prob;
            source_correct[ srccoord ]   = ( dcol<5 ) ? 1 : 0;
            source_max_errflow[ srccoord ] = fabs( tarcol-truecol ); // error in predicted flow            
          }
        }
        
        //hsrc->Fill( srccol, tick );
        if ( prob>=1.0 ) prob = 0.9999;
        // std::cout << "(" << dcol << ": " << prob << ") srccol=" << srccol << " (" << hit.srcwire << ") "
        //           << " tick=" << tick << " row=" << row
        //           << " targetcol=" << tarcol
        //           << " truecol=" << truecol
        //           << " flow=" << trueflow
        //           << std::endl;
        hprob_v_coldist->Fill( dcol, prob );
      }//end of loop over hits

      for ( auto it=source_max_errflow.begin(); it!=source_max_errflow.end(); it++ ) {
        herrflow[iflow]->Fill( it->second );
      }

      const larflow::FlowMatchMap& flowmap = matchmap_v[iflow];
      std::cout << "check: "
                << " sparse source image pixels=" <<  sparse_src.len()
                << " flowmap pixels=" << flowmap.nsourceIndices()
                << std::endl;
      
      assert( sparse_src.len()==flowmap.nsourceIndices() );

      for (int iidx=0; iidx<flowmap.nsourceIndices(); iidx++) {
        auto const& target_indices = flowmap.getTargetIndices(iidx);
        int ntargets = target_indices.size();
        hnmatches[iflow]->Fill( (float)ntargets );

        int srccol = (int)sparse_src.getfeature( iidx, 1 );
        int srcrow = (int)sparse_src.getfeature( iidx, 0 );
        std::array<int,2> srccoord = { srccol, srcrow };
        
        if ( source_max_score.find( srccoord )!=source_max_score.end() ) {
          if ( source_correct[ srccoord ]==1 )
            hprob_v_nmatches_good[iflow]->Fill( (float)ntargets, source_max_score[ srccoord ] );
          else
            hprob_v_nmatches_bad[iflow]->Fill( (float)ntargets, source_max_score[ srccoord ] );            
        }
        
      }//end of loop over source pixels via index number
      
    }//end of loop over flow loop
    
  }//event loop

  //hprob_v_coldist->Write();
  outfile->Write();
  outfile->Close();
  
  io.finalize();
  llio.close();
  
  std::cout << "FIN" << std::endl;
  return 0;
};
