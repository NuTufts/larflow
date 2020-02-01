#include "PrepMatchTriplets.h"
#include "FlowTriples.h"

#include <sstream>
#include <ctime>

namespace larflow {


  void PrepMatchTriplets::process( const std::vector<larcv::Image2D>& adc_v,
                                   const std::vector<larcv::Image2D>& badch_v,
                                   const float adc_threshold ) {

    std::clock_t start = std::clock();
    
    // first we make a common sparse image
    _sparseimg_vv = larflow::FlowTriples::make_initial_sparse_image( adc_v, adc_threshold );
      

    // then we make the flow triples for each of the six flows
    std::vector< larflow::FlowTriples > triplet_v( larflow::kNumFlows );
    for (int sourceplane=0; sourceplane<(int)adc_v.size(); sourceplane++ ) {
      for (int targetplane=0; targetplane<(int)adc_v.size(); targetplane++ ) {

        if ( sourceplane==targetplane ) continue;

        int flowindex = (int)larflow::LArFlowConstants::getFlowDirection( sourceplane, targetplane );
        // save pixel position, rather than index: we will reindex after adding dead channels
        triplet_v[ flowindex ] = FlowTriples( sourceplane, targetplane,
                                              adc_v, badch_v,
                                              _sparseimg_vv, 10.0, false );        
      }
    }

    // collect the unique dead channel additions to each plane
    std::set< std::pair<int,int> > deadpixels_to_add[ adc_v.size() ];

    // add the unique dead channel additions
    std::cout << "sparse pixel totals before deadch additions: "
              << "(" << _sparseimg_vv[0].size() << "," << _sparseimg_vv[1].size() << "," << _sparseimg_vv[2].size() << ")"
              << std::endl;
    
    for ( auto& triplet : triplet_v ) {

      int otherplane = triplet.get_other_plane_index();
      
      std::vector<FlowTriples::PixData_t>& pix_v = triplet.getDeadChToAdd()[ otherplane ];
      
      for ( auto& pix : pix_v ) {
        auto it = deadpixels_to_add[ otherplane ].find( std::pair<int,int>( pix.row, pix.col ) );
        if ( it==deadpixels_to_add[ otherplane ].end() ) {
          // unique, add to sparse image
          pix.val = 1.0;
          _sparseimg_vv[otherplane].push_back( pix );
        }
      }
    }

    std::cout << "sparse pixel totals before deadch additions: "
              << "(" << _sparseimg_vv[0].size() << "," << _sparseimg_vv[1].size() << "," << _sparseimg_vv[2].size() << ")"
              << std::endl;
    
    // sort all pixels
    for ( auto& pix_v : _sparseimg_vv ) {
      std::sort( pix_v.begin(), pix_v.end() );
    }

    // condense and reindex matches
    std::set< std::vector<int> > triplet_set;
    _triplet_v.clear();
    _triplet_v.reserve( 500000  );
    
    for ( auto& triplet_data : triplet_v ) {    
      int srcplane = triplet_data.get_source_plane_index();
      int tarplane = triplet_data.get_target_plane_index();
      int othplane = triplet_data.get_other_plane_index();
      for ( auto& trip : triplet_data.getTriples() ) {

        std::vector<FlowTriples::PixData_t> pix_v(3);
        pix_v[ srcplane ] = FlowTriples::PixData_t( trip[3], trip[0], 0.0 );
        pix_v[ tarplane ] = FlowTriples::PixData_t( trip[3], trip[1], 0.0 );
        pix_v[ othplane ] = FlowTriples::PixData_t( trip[3], trip[2], 0.0 );

        auto it_src = std::lower_bound( _sparseimg_vv[srcplane].begin(), _sparseimg_vv[srcplane].end(), pix_v[ srcplane ] );
        auto it_tar = std::lower_bound( _sparseimg_vv[tarplane].begin(), _sparseimg_vv[tarplane].end(), pix_v[ tarplane ] );
        auto it_oth = std::lower_bound( _sparseimg_vv[othplane].begin(), _sparseimg_vv[othplane].end(), pix_v[ othplane ] );

        if ( it_src==_sparseimg_vv[srcplane].end()
             || it_tar==_sparseimg_vv[tarplane].end()
             || it_oth==_sparseimg_vv[othplane].end() ) {
          std::stringstream ss;
          ss << "Did not find one of sparse image pixels for col triplet=(" << trip[0] << "," << trip[1] << "," << trip[2] << ")";
          ss << " found-index=("
             << it_src-_sparseimg_vv[srcplane].begin() << ","
             << it_tar - _sparseimg_vv[tarplane].begin() << ","
             << it_oth - _sparseimg_vv[othplane].begin() << ")"
             << std::endl;
          throw std::runtime_error( ss.str() );
        }

        std::vector<int> imgcoord_v(4);
        imgcoord_v[ srcplane ] = trip[0];
        imgcoord_v[ tarplane ] = trip[1];
        imgcoord_v[ othplane ] = trip[2];
        imgcoord_v[ 3 ]        = trip[3];
        auto it_trip = triplet_set.find( imgcoord_v );
        if ( it_trip==triplet_set.end() ) {
          triplet_set.insert( imgcoord_v );

          std::vector<int> imgindex_v(4);
          imgindex_v[ srcplane ] = it_src - _sparseimg_vv[srcplane].begin();
          imgindex_v[ tarplane ] = it_tar - _sparseimg_vv[tarplane].begin();
          imgindex_v[ othplane ] = it_oth - _sparseimg_vv[othplane].begin();
          imgindex_v[ 3 ]        = trip[3];

          _triplet_v.push_back( imgcoord_v );
        }
        
      }

    }

    std::clock_t end = std::clock();
    std::cout << "[PrepMatchTriplets] made total of " << _triplet_v.size()
              << " unique index triplets. time elapsed=" << float(end-start)/float(CLOCKS_PER_SEC)
              << std::endl;

  }//end of process method

  std::vector<TH2D> PrepMatchTriplets::plot_sparse_images( const std::vector<larcv::Image2D>& adc_v,
                                                           std::string hist_stem_name )
  {
    std::vector<TH2D> out_v;
    for ( int p=0; p<(int)adc_v.size(); p++ ) {
      std::stringstream ss;
      ss << "htriples_plane" << p << "_" << hist_stem_name;
      auto const& meta = adc_v[p].meta();
      TH2D hist( ss.str().c_str(), "",
                 meta.cols(), meta.min_x(), meta.max_x(),
                 meta.rows(), meta.min_y(), meta.max_y() );

      for ( auto const& pix : _sparseimg_vv[p] ) {
        hist.SetBinContent( pix.col+1, pix.row+1, pix.val );
      }
      
      out_v.emplace_back(std::move(hist));
    }
    return out_v;
  }
  
}
