#include "PrepFlowMatchData.hh"

#include "core/LArUtil/Geometry.h"

#include "larcv/core/DataFormat/EventSparseImage.h"
#include "larcv/core/DataFormat/SparseImage.h"
#include "larcv/core/DataFormat/EventImage2D.h"

#include <sstream>

namespace larflow {

  // ---------------------------------------------------------
  // FLOW MATCH MAP CLASS
  
  void FlowMatchMap::add_matchdata( int src_index,
                                    const std::vector<int>& target_indices,
                                    const std::vector<int>& truth_v ) {

    if ( truth_v.size()!=target_indices.size() ) {
      throw std::runtime_error( "truth and target index vectors not the same size" );
    }
    
    _target_map[src_index] = target_indices;
    _truth_map[src_index]  = truth_v;
    
  }

  const std::vector<int>& FlowMatchMap::getTargetIndices( int src_index ) const {

    auto it = _target_map.find( src_index );
    if ( it==_target_map.end() ) {
      std::stringstream msg;
      msg << "did not find source index=" << src_index << ".";
      throw std::runtime_error( msg.str() );
    }

    return it->second;
  }

  const std::vector<int>& FlowMatchMap::getTruthVector( int src_index ) const {

    auto it = _truth_map.find( src_index );
    if ( it==_truth_map.end() ) {
      std::stringstream msg;
      msg << "did not find source index=" << src_index << ".";
      throw std::runtime_error( msg.str() );
    }
    
    return it->second;
  }
  
  // ---------------------------------------------------------
  // FLOW MATCH MAP CLASS FACTORY
  
  static PrepFlowMatchDataFactory __global_PrepFlowMatchDataFactory__;
  
  // ---------------------------------------------------------
  // FLOW MATCH MAP CLASS

  void PrepFlowMatchData::configure( const larcv::PSet& pset ) {

    _input_adc_producername      = pset.get<std::string>("InputADC",      "wire");
    _input_trueflow_producername = pset.get<std::string>("InputTrueFlow", "larflow" );
    
  }

  void PrepFlowMatchData::initialize() {
    _setup_ana_tree();
    _extract_wire_overlap_bounds();
  }
  
  bool PrepFlowMatchData::process( larcv::IOManager& mgr ) {
    // first we sparsify data,
    // then for each flow direction,
    //   we loop through each source pixel
    //   and make a list of target image indices that the source pixel could possibly flow to.
    //   we then provide a {0,1} label for each possbile flow, with 1 reserved for the correct match
    // we also need to provide a weight for each event for bad and good matches, to balance that choice.

    auto ev_adc  = (larcv::EventImage2D*)mgr.get_data(larcv::kProductImage2D,_input_adc_producername);
    auto ev_flow = (larcv::EventImage2D*)mgr.get_data(larcv::kProductImage2D,_input_trueflow_producername);
    LARCV_DEBUG() << "get adc and flow images. len(adc)=" << ev_adc->Image2DArray().size()
                  << " len(flow)=" << ev_flow->Image2DArray().size() << std::endl;

    // make sparse image for the source ADC + 2 x (true flow + matachabilitity)+weights+nchoice,
    //  + 2 x sparse target images

    // goes into function eventually
    int srcindex        = 2;     // Y
    int target_index[2] = {0,1}; // Y->U, Y->V
    int flow_index[2]   = {4,5}; // Y->U, Y->V

    auto const& srcimg = ev_adc->Image2DArray().at(srcindex);
    const larcv::Image2D* tarimg[2] = { &ev_adc->Image2DArray().at(target_index[0]),
                                        &ev_adc->Image2DArray().at(target_index[1]) };

    // create matchability image
    std::vector<larcv::Image2D> matchability_v;
    for ( size_t i=0; i<2; i++ ) {

      LARCV_DEBUG() << "make matchability image for flowdir=" << i << std::endl;
      
      larcv::Image2D matchability( srcimg.meta() );
      matchability.paint(0.0);

      auto const& tar     = *tarimg[i];
      auto const& flowimg = ev_flow->Image2DArray().at( flow_index[i] );
      
      // we check matchability of flow.  does flow go into a dead region?
      for ( size_t c=0; c<srcimg.meta().cols(); c++ ) {
        for ( size_t r=0; r<srcimg.meta().rows(); r++ ) {
          float adc = srcimg.pixel(r,c);
          if ( adc<10.0 ) continue;
          float flow = flowimg.pixel(r,c);

          if ( flow<=-4000) continue;
          
          int target_col = (int)c + (int)flow;
          
          int hastarget = 1;
          if ( target_col<0 || target_col>=(int)tar.meta().cols() )
            hastarget = 0;
          else if ( tar.pixel( r, target_col )<10.0 )
            hastarget = 0;
          
          if ( hastarget==1 )
            matchability.set_pixel(r,c,1.0);
          
        }
      }//end of col loop

      matchability_v.emplace_back( std::move(matchability) );
    }

    // Now make sparse images

    // Source Image: combined with features [adc,trueflow1,trueflow2,matchable1,matchable2]
    std::vector< const larcv::Image2D* > img_v;
    std::vector< float > threshold_v = { 10.0, -3999.0, -3999.0, -1, -1 }; // threshold value to include pixel
    std::vector< int >   cuton_v     = {    1,       0,       0,  0,  0 }; // feature to make cut on (COMBINED USING OR)

    LARCV_DEBUG() << "make source sparse image" << std::endl;

    img_v.push_back( &srcimg );
    img_v.push_back( &ev_flow->Image2DArray().at( flow_index[0] ) );
    img_v.push_back( &ev_flow->Image2DArray().at( flow_index[1] ) );
    img_v.push_back( &matchability_v[0] );
    img_v.push_back( &matchability_v[1] );    

    larcv::SparseImage spsrc( img_v, threshold_v, cuton_v ); //make the sparse image

    LARCV_DEBUG() << "Number of source image pixels: " << spsrc.len() << std::endl;
    LARCV_DEBUG() << "Occupancy of source image: "
                  << float(spsrc.len())/float(spsrc.meta(0).cols()*spsrc.meta(1).rows())
                  << std::endl;    

    std::vector< larcv::SparseImage > spimg_v;
    spimg_v.emplace_back( std::move(spsrc) );

    // sparse image for the target images. we just need the adc values
    for (int i=0; i<2; i++ ) {
      LARCV_DEBUG() << "make target sparse image: flowdir=" << i << std::endl;
    
      std::vector< const larcv::Image2D* > imgtar_v;
      imgtar_v.push_back( tarimg[i] );
      std::vector<float> tar_threshold_v(1,10.0);
      std::vector<int>   tar_cuton_v(1,1);
      larcv::SparseImage sptar( imgtar_v, tar_threshold_v, tar_cuton_v );
      LARCV_DEBUG() << "target (flowdir=" << i << ") image pixels: " << sptar.len() << std::endl;
      spimg_v.emplace_back( std::move(sptar) );
    }
    
    // now we make the flowmatch map for the source image, for both flows.
    _matchdata_v->clear();
    for (int i=0; i<2; i++ ) {

      LARCV_DEBUG() << "make match map: flowdir=" << i << std::endl;
      
      FlowMatchMap matchmap;
      auto& sparsesrc = spimg_v[0];
      auto& sptar     = spimg_v[1+i];
      
      // first, we scan the target sparse image, making a map of row to vector of column indices
      std::map< int, std::vector<int> > targetmap;
      for ( size_t ipt=0; ipt<sptar.len(); ipt++ ) {
        int   col = (int)sptar.getfeature(ipt,1);
        int   row = (int)sptar.getfeature(ipt,0);
        float adc = sptar.getfeature(ipt,2);

        auto it=targetmap.find( row );
        if ( it==targetmap.end() ) {
          targetmap[row] = std::vector<int>();
          it = targetmap.find(row);
        }
        it->second.push_back( ipt ); // add in the index of this point
      }
      LARCV_DEBUG() << "target[row] -> {target indices} map define (flowdir=" << i << ")."
                    << " elements=" << targetmap.size()
                    << std::endl;

      int npix_w_matches  = 0;
      int npix_wo_matches = 0;
      
      // now we can make the src pixel to target pixel choices map
      for ( int ipt=0; ipt<sparsesrc.len(); ipt++ ) {
        
        int   col = (int)sparsesrc.getfeature(ipt,1);
        int   row = (int)sparsesrc.getfeature(ipt,0);
        int   matchable = (int)sparsesrc.getfeature(ipt,5+i);

        float umin = _wire_bounds[i][col][0];
        float umax = _wire_bounds[i][col][1];

        std::vector<int> matchable_target_cols;
        
        auto it=targetmap.find(row);
        if ( it!=targetmap.end() ) {
          auto const& target_index_v = targetmap[row];

          std::vector< int > truth_v;
          std::vector< int > within_bounds_target_v;

          // find the true one
          float flow = sparsesrc.getfeature(ipt,3+i); // the true flow
          int target_col = col + (int)flow;

          int nmatches = 0;
          std::vector<int> tar_col_v;
          for ( size_t idx=0; idx<target_index_v.size(); idx++ ) {
            int tarcol = sptar.getfeature( target_index_v[idx], 1 );

            if ( tarcol<(int)umin || tarcol>(int)umax ) continue;
            
            tar_col_v.push_back( tarcol );
            within_bounds_target_v.push_back( target_index_v[idx] );
            if ( tarcol==target_col ) {
              truth_v.push_back(1);
              nmatches++;
            }
            else {
              truth_v.push_back(0);
            }
          }
          if ( matchable==1 ) {
            if ( nmatches!=1 )
              LARCV_CRITICAL() << "did not find matchable column" << std::endl;

            if ( nmatches==1 )
              npix_w_matches++;
            else
              npix_wo_matches++;
          }

          std::stringstream sstarcol;          
          if ( logger().debug() ) {
            sstarcol << "{ ";
            for ( auto const& tarcol : tar_col_v ) sstarcol << tarcol << " ";
            sstarcol << "}";
          }
          
          // LARCV_DEBUG() << "srcpixel[" << ipt << ", (" << row << "," << col << ")] "
          //               << " flow (" << flow << ") to targetcol=" << target_col
          //               << " matchable=" << matchable
          //               << " bounds=[" << umin << "," << umax << "]"
          //               << std::endl;
          // LARCV_DEBUG() << "  has " << target_index_v.size() << " potential matches " << sstarcol.str()
          //               << "  and " << nmatches << " correct match" << std::endl;
          matchmap.add_matchdata( ipt, within_bounds_target_v, truth_v );
        }
        else {
          matchmap.add_matchdata( ipt, std::vector<int>(), std::vector<int>() );
        }
      }//loop over points

      _matchdata_v->emplace_back( std::move(matchmap) );
      LARCV_INFO() << "matched map flowdir=" << i << ": matchable hasmatch=" << npix_w_matches << " hasnomatch=" << npix_wo_matches << std::endl;
      
    }//end of loop over flow directions

    LARCV_DEBUG() << "pass sparse images to iomanager" << std::endl;
    larcv::EventSparseImage* ev_out = (larcv::EventSparseImage*)mgr.get_data(larcv::kProductSparseImage, "larflow" );
    ev_out->Emplace( std::move( spimg_v ) );

    LARCV_DEBUG() << "save flow match maps to ana tree" << std::endl;    
    _ana_tree->Fill();

    LARCV_DEBUG() << "done" << std::endl;    
    return true;
  }

  void PrepFlowMatchData::finalize() {
    _ana_tree->Write();
  }

  void PrepFlowMatchData::_setup_ana_tree() {
    LARCV_DEBUG() << "create tree, make container, set branch" << std::endl;
    _ana_tree = new TTree("flowmatchdata","Provides map from source to target pixels to match");
    _matchdata_v = new std::vector< FlowMatchMap >();
    _ana_tree->Branch( "matchmap", _matchdata_v );
  }

  void PrepFlowMatchData::_extract_wire_overlap_bounds() {

    const larutil::Geometry* geo = larutil::Geometry::GetME();
    
    for (int i=0; i<2; i++ ) {
      LARCV_DEBUG() << "defined overlapping wire bounds for Y->plane[" << i << "]" << std::endl;
      _wire_bounds[i].clear();

      for (int isrc=0; isrc<(int)geo->Nwires(2); isrc++ ) {
        Double_t xyzstart[3];
        Double_t xyzend[3];
        geo->WireEndPoints( (UChar_t)2, (UInt_t)isrc, xyzstart, xyzend );

        float u1 = geo->WireCoordinate( xyzstart, (UInt_t)i );
        float u2 = geo->WireCoordinate( xyzend,   (UInt_t)i );

        float umin = (u1<u2) ? u1 : u2;
        float umax = (u1>u2) ? u1 : u2;

        umin -= 5.0;
        umax += 5.0;

        if ( umin<0 ) umin = 0;
        if ( umax<0 ) umax = 0;

        if ( (int)umin>=geo->Nwires(i) ) umin = (float)geo->Nwires(i)-1;
        if ( (int)umax>=geo->Nwires(i) ) umax = (float)geo->Nwires(i)-1;

        _wire_bounds[i][isrc] = std::vector<int>{ (int)umin, (int)umax };
        //LARCV_DEBUG() << " src[" << isrc << "] -> plane[" << i << "]: (" << umin << "," << umax << ")" << std::endl;
      }
    }
    LARCV_DEBUG() << "defined overlapping wire bounds" << std::endl;
    //std::cin.get();
  }

}
