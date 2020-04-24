#include "ShowerLikelihoodBuilder.h"

#include "larcv/core/DataFormat/DataFormatTypes.h"
#include "larcv/core/DataFormat/EventImage2D.h"
#include "DataFormat/larflow3dhit.h"
#include "DataFormat/mcshower.h"
#include "LArUtil/SpaceChargeMicroBooNE.h"
#include "ublarcvapp/MCTools/MCPixelPGraph.h"

namespace larflow {
namespace reco {

  ShowerLikelihoodBuilder::ShowerLikelihoodBuilder()
  {
    _hll = new TH2F("lfshower_ll","",2000, -10, 190, 1000, 0, 100 );
    _psce = new larutil::SpaceChargeMicroBooNE();
  }

  ShowerLikelihoodBuilder::~ShowerLikelihoodBuilder()
  {
    // if ( _hll ) delete _hll;
    // _hll = nullptr;
  }
  
  void ShowerLikelihoodBuilder::process( larcv::IOManager& iolcv, larlite::storage_manager& ioll )
  {

    // steps.
    //
    // first we need to assemble true triplet points
    // we start by masking out adc images by segment image
    // then we pass the masked image into the triplet proposal algorithm
    // we filter out the proposals by true match + source pixel being on an electron (true ssnet label)
    //
    // after ground truth points are made, we can build the calculations we want


    // build true triplets
    larcv::EventImage2D* ev_adc      =
      (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, "wire" );
    larcv::EventImage2D* ev_seg      =
      (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, "segment" );
    larcv::EventImage2D* ev_trueflow =
      (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, "larflow" );

    larlite::event_mcshower* ev_mcshower
      = (larlite::event_mcshower*)ioll.get_data( larlite::data::kMCShower, "mcreco" );

    std::vector<larcv::Image2D> masked_v;
    std::vector<larcv::Image2D> badch_v;
    for ( size_t p=0; p<3; p++ ) {
      auto const& adc = ev_adc->Image2DArray()[p];
      auto const& seg = ev_seg->Image2DArray()[p];

      larcv::Image2D masked(adc.meta());
      masked.paint(0.0);

      for ( size_t r=0; r<adc.meta().rows(); r++) {
        for ( size_t c=0; c<adc.meta().cols(); c++) {

          int pid = (int)seg.pixel(r,c);          
          if ( pid !=larcv::kROIEminus && pid!=larcv::kROIGamma )
            continue;

          // bool found = false;
          // for (int dr=-1; dr<=1; dr++) {
          //   for (int dc=-1; dc<=1; dc++) {

          //     int row = (int)r+dr;
          //     int col = (int)c+dc;
              
          //     if ( row<0 || col<0 ) continue;
          //     if ( row>=(int)adc.meta().rows() || col>=(int)adc.meta().cols() ) continue;
              
          //     int pid = (int)seg.pixel(row,col);
          //     if ( pid ==larcv::kROIEminus || pid==larcv::kROIGamma ) {
          //       found = true;
          //     }
          //     if ( found ) break;
          //   }
          //   if ( found ) break;
          // }
          
          // copy pixel
          if ( adc.pixel(r,c)>10 )
            masked.set_pixel( r, c, adc.pixel(r,c) );
          else
            masked.set_pixel( r, c, 15.0 );
        }
      }
      masked_v.emplace_back( std::move(masked) );

      larcv::Image2D badch(adc.meta());
      badch.paint(50.0);
      badch_v.emplace_back( std::move(badch) );
    }

    // images made, now we make the triplets
    tripletalgo.process( masked_v, badch_v, 1.0, true );
    tripletalgo.make_truth_vector( ev_trueflow->Image2DArray() );

    // lets save the true triplets as larflow3dhits
    
    std::vector<larlite::larflow3dhit> truehit_v;
    

    for ( size_t i=0; i<tripletalgo._triplet_v.size(); i++ ) {

      if ( tripletalgo._truth_v[i]!=1 )
        continue;
      
      larlite::larflow3dhit hit;
      // * [0-2]:   x,y,z
      // * [3-9]:   6 flow direction scores + 1 max scire (deprecated based on 2-flow paradigm. for triplet, [8] is the only score stored 
      // * [10-12]: 3 ssnet scores, (bg,track,shower)
      // * [13]:    1 keypoint label score

      hit.resize( 14, 0 );
      for (size_t v=0; v<3; v++ ) hit[v] = tripletalgo._pos_v[i][v];
      hit[12] = 1.0;
      hit[13] = 0.0;
      hit.tick = tripletalgo._triplet_v[i][3];
      hit.srcwire = tripletalgo._triplet_v[i][2];
      hit.targetwire.resize(3);
      
      
      for (size_t p=0; p<3; p++)
        hit.targetwire[p] = tripletalgo._sparseimg_vv[p][ tripletalgo._triplet_v[i][p] ].col;
      hit.idxhit = i;

      truehit_v.emplace_back( std::move(hit) );
        
    }

    // now we parse the truth, by building the particle graph
    ublarcvapp::mctools::MCPixelPGraph mcpg;
    mcpg.set_adc_treename( "wire" );
    mcpg.buildgraph( iolcv, ioll );

    // look for the primary electron
    std::vector<ublarcvapp::mctools::MCPixelPGraph::Node_t*> nodes_v
      = mcpg.getPrimaryParticles();

    // get the shower direction
    bool found_electron = false;
    std::vector<float> shower_dir(3,0);
    std::vector<float> shower_vtx(3,0);
    float dirnorm = 0.;
    
    for (auto& pnode : nodes_v ) {
      if ( pnode->type!=1 || abs(pnode->pid)!=11 )
        continue;

      auto const& mcsh = ev_mcshower->at( pnode->vidx );

      shower_dir[0] = mcsh.Start().Px();
      shower_dir[1] = mcsh.Start().Py();
      shower_dir[2] = mcsh.Start().Pz();
      shower_vtx[0] = mcsh.Start().X();
      shower_vtx[1] = mcsh.Start().Y();
      shower_vtx[2] = mcsh.Start().Z();
      
      for (int v=0; v<3; v++)
        dirnorm += shower_dir[v]*shower_dir[v];
      dirnorm = sqrt(dirnorm);
      for (int v=0; v<3; v++)
        shower_dir[v] /= dirnorm;
      found_electron  = true;
      break;
    }


    std::cout << "found electron: " << found_electron << std::endl;

    std::vector<float> shower_end(3,0);
    for (int p=0; p<3; p++ ) shower_end[p] = shower_vtx[p] + 3.0*shower_dir[p];

    std::vector<double> offset = _psce->GetPosOffsets( shower_vtx[0], shower_vtx[1], shower_vtx[2] );
    std::vector<float> shower_vtx_sce  = { shower_vtx[0] - (float)offset[0] + 0.6,
                                           shower_vtx[1] + (float)offset[1],
                                           shower_vtx[2] + (float)offset[2] };
    offset = _psce->GetPosOffsets( shower_end[0], shower_end[1], shower_end[2] );
    std::vector<float> shower_end_sce  = { shower_end[0] - (float)offset[0] + 0.6,
                                           shower_end[1] + (float)offset[1],
                                           shower_end[2] + (float)offset[2] };
    // adjust shower dir due to SCE
    float scenorm = 0.;
    for (int i=0; i<3; i++ ) {
      shower_dir[i] = shower_end_sce[i]-shower_vtx_sce[i];
      scenorm += shower_dir[i]*shower_dir[i];
    }
    scenorm = sqrt(scenorm);
    for (int i=0; i<3; i++ ) shower_dir[i] /= scenorm;
    
    // convert hits into cluster_t objects
    
    
    // fill profile histogram

    // break into clusters
    // truth ID the trunk cluster
    // save vars for the trunk verus non-trunk clutsters
    // 
    _fillProfileHist( truehit_v, shower_dir, shower_vtx_sce );

    mcpg.printGraph();

    larlite::event_larflow3dhit* evout = (larlite::event_larflow3dhit*)ioll.get_data(larlite::data::kLArFlow3DHit, "trueshowerhits" );
    for (auto& hit : truehit_v )
      evout->emplace_back( std::move(hit) );

    larcv::EventImage2D* evimgout = (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D, "trueshoweradc" );
    evimgout->Emplace( std::move(masked_v) );
    
  }

  void ShowerLikelihoodBuilder::_fillProfileHist( const std::vector<larlite::larflow3dhit>& truehit_v,
                                                  std::vector<float>& shower_dir,
                                                  std::vector<float>& shower_vtx )
  {

    // get distance of point from pca-axis
    // http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html

    std::cout << "Fill hits for shower[ "
              << "dir=(" << shower_dir[0] << "," << shower_dir[1] << "," << shower_dir[2] << ") "
              << "vtx=(" << shower_vtx[0] << "," << shower_vtx[1] << "," << shower_vtx[2] << ") "
              << "] with nhits=" << truehit_v.size()
              << std::endl;

    std::vector<float> shower_end(3);
    std::vector<float> d3(3);
    float len3 = 1.0;
    for (int i=0; i<3; i++ ) {
      shower_end[i] = shower_vtx[i] + shower_dir[i];
      d3[i] = shower_dir[i];
    }
      
    for ( auto const& hit : truehit_v ) {
      
      std::vector<float> pt = { hit[0], hit[1], hit[2] };

      std::vector<float> d1(3);
      std::vector<float> d2(3);

      float len1 = 0.;
      for (int i=0; i<3; i++ ) {
        d1[i] = pt[i] - shower_vtx[i];
        d2[i] = pt[i] - shower_end[i];
        len1 += d1[i]*d1[i];
      }
      len1 = sqrt(len1);

      // cross-product
      std::vector<float> d1xd2(3);
      d1xd2[0] =  d1[1]*d2[2] - d1[2]*d2[1];
      d1xd2[1] = -d1[0]*d2[2] + d1[2]*d2[0];
      d1xd2[2] =  d1[0]*d2[1] - d1[1]*d2[0];
      float len1x2 = 0.;
      for ( int i=0; i<3; i++ ) {
        len1x2 += d1xd2[i]*d1xd2[i];
      }
      len1x2 = sqrt(len1x2);
      float rad  = len1x2/len3; // distance of point from PCA-axis

      float proj = 0.;
      for ( int i=0; i<3; i++ )
        proj += shower_dir[i]*d1[i];

      // std::cout << "  hit: (" << pt[0] <<  "," << pt[1] << "," << pt[2] << ") "
      //           << " dist=" << len1
      //           << " proj=" << proj
      //           << " rad=" << rad
      //           << std::endl;
      float w=1.;
      if ( rad>0 ) w = 1.0/rad;
      _hll->Fill( proj, rad, w );
    }
    
  }

}
}
