#include "PrepFlowMatchData.hh"

#include "core/LArUtil/Geometry.h"

#include "larcv/core/DataFormat/EventSparseImage.h"
#include "larcv/core/DataFormat/SparseImage.h"
#include "larcv/core/DataFormat/EventImage2D.h"
#include "larcv/core/DataFormat/EventChStatus.h"

#include <sstream>

namespace larflow {

  
  // ---------------------------------------------------------
  // PREP FLOW MATCH MAP CLASS FACTORY
  
  static PrepFlowMatchDataFactory __global_PrepFlowMatchDataFactory__;
  
  // ---------------------------------------------------------
  // PREP FLOW MATCH MAP CLASS

  PrepFlowMatchData::PrepFlowMatchData( std::string instance_name )
    : larcv::ProcessBase(instance_name),
    _input_adc_producername(""),
    _input_trueflow_producername(""),
    _has_mctruth(false),
    _use_ana_tree(false),
    _use_soft_truth(false),
    _positive_example_distance(0),
    _use_3plane_constraint(true),
    _debug_detailed_output(false),
    _source_plane(-1),
    _matchdata_v(nullptr),
    _ana_tree(nullptr)
  {}

  void PrepFlowMatchData::configure( const larcv::PSet& pset ) {

    // the source plane index
    _source_plane                = pset.get<int>("SourcePlane");

    // the tree holding the ADC images
    _input_adc_producername      = pset.get<std::string>("InputADC", "wire");

    // the tree holding larcv::ChStatus
    _input_chstatus_producername = pset.get<std::string>("InputChStatus", "wire");
    
    // the tree holding the true flow images
    _input_trueflow_producername = pset.get<std::string>("InputTrueFlow", "larflow" );

    // [not implemented]: truth score is vector with intermediate scores, not a one-hot vector
    _use_soft_truth              = pset.get<bool>("UseSoftTruthVector",false);

    // has MC truth information
    _has_mctruth                 = pset.get<bool>("HasMCTruth",false);

    // distance witin which labeled as truth
    _positive_example_distance   = pset.get<int>("PositiveExampleDistance",5);

    // only keep two-plane candidate matches where corresponding pixel in other wire has charge or is in dead region
    _use_3plane_constraint       = pset.get<bool>("Use3PlaneConstraint",true);

    _debug_detailed_output       = pset.get<bool>("DetailedDebugOutput",false);

    useAnaTree(true);
  }

  void PrepFlowMatchData::initialize() {
    _setup_ana_tree();
    _extract_wire_overlap_bounds();
  }

  const std::vector<FlowMatchMap>& PrepFlowMatchData::getMatchData() const {
    if ( _matchdata_v==nullptr ) {
      LARCV_CRITICAL() << "Need to initialize class first." << std::endl;
      throw std::runtime_error("Need to initialize first");
    }
    return *_matchdata_v;
  }
  
  bool PrepFlowMatchData::process( larcv::IOManager& mgr ) {
    // first we sparsify data,
    // then for each flow direction,
    //   we loop through each source pixel
    //   and make a list of target image indices that the source pixel could possibly flow to.
    //   we then provide a {0,1} label for each possbile flow, with 1 reserved for the correct match
    // we also need to provide a weight for each event for bad and good matches, to balance that choice.

    larcv::EventImage2D* ev_adc  = (larcv::EventImage2D*)mgr.get_data(larcv::kProductImage2D,_input_adc_producername);
    LARCV_DEBUG() << "get adc and flow images. len(adc)=" << ev_adc->Image2DArray().size() << std::endl;
    
    larcv::EventImage2D* ev_flow = nullptr;
    if ( _has_mctruth ) {
      ev_flow = (larcv::EventImage2D*)mgr.get_data(larcv::kProductImage2D,_input_trueflow_producername);
      LARCV_DEBUG() << " len(flow)=" << ev_flow->Image2DArray().size() << std::endl;
    }

    larcv::EventChStatus* ev_chstatus = nullptr;
    if ( _use_3plane_constraint ) {
      ev_chstatus = (larcv::EventChStatus*)mgr.get_data(larcv::kProductChStatus, _input_chstatus_producername );
    }

    // make sparse image for the source ADC + 2 x (true flow + matachabilitity)+weights+nchoice,
    //  + 2 x sparse target images

    // goes into function eventually
    int srcindex        = _source_plane;
    int target_index[2] = { _target_planes[_flowdirs[0]],
                            _target_planes[_flowdirs[1]] };
    int flow_index[2]   = { (int)_flowdirs[0], (int)_flowdirs[1] }; // Y->U, Y->V

    auto const& srcimg = ev_adc->Image2DArray().at(srcindex);
    const larcv::Image2D* tarimg[2] = { &ev_adc->Image2DArray().at(target_index[0]),
                                        &ev_adc->Image2DArray().at(target_index[1]) };

    // create matchability image
    std::vector<larcv::Image2D> matchability_v;
    if ( _has_mctruth )  {
      for ( size_t i=0; i<2; i++ ) {

        LARCV_DEBUG() << "make matchability image for flowdir=" << i << std::endl;
      
        larcv::Image2D matchability( srcimg.meta() );
        matchability.paint(0.0);
        
        auto const& tar     = *tarimg[i];
        auto const& flowimg = ev_flow->Image2DArray().at( flow_index[i] );
        
        // we check matchability of flow.  does flow go into a dead region?
        for ( size_t c=0; c<srcimg.meta().cols(); c++ ) {
          for ( size_t r=0; r<srcimg.meta().rows(); r++ ) {
            float adc = srcimg.pixel(r,c,__FILE__,__LINE__);
            if ( adc<10.0 ) continue;
            float flow = flowimg.pixel(r,c,__FILE__,__LINE__);
            
            if ( flow<=-4000) continue;
            
            int target_col = (int)c + (int)flow;
            
            int hastarget = 1;
            if ( target_col<0 || target_col>=(int)tar.meta().cols() )
              hastarget = 0;
            else if ( tar.pixel( r, target_col,__FILE__,__LINE__ )<10.0 )
              hastarget = 0;
            
            if ( hastarget==1 )
              matchability.set_pixel(r,c,1.0);
            
          }
        }//end of col loop
        
        matchability_v.emplace_back( std::move(matchability) );
      }
    }//end of has truth

    // Now make sparse images

    // Source Image: combined with features [adc,trueflow1,trueflow2,matchable1,matchable2]
    std::vector< const larcv::Image2D* > img_v;
    std::vector< float > threshold_v; 
    std::vector< int >   cuton_v;     

    if ( _has_mctruth ) {
      threshold_v = { 10.0, -3999.0, -3999.0, -1, -1 }; // threshold value to include pixel
      cuton_v     = {    1,       0,       0,  0,  0 }; // feature to make cut on (COMBINED USING OR)
    }
    else {
      threshold_v = { 10.0 };
      cuton_v     = { 1 };
    }

    LARCV_DEBUG() << "make source sparse image" << std::endl;

    img_v.push_back( &srcimg );
    if ( _has_mctruth ) {
      img_v.push_back( &ev_flow->Image2DArray().at( flow_index[0] ) );
      img_v.push_back( &ev_flow->Image2DArray().at( flow_index[1] ) );
      img_v.push_back( &matchability_v[0] );
      img_v.push_back( &matchability_v[1] );
    }

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

      int source_plane = _source_plane;
      int target_plane = target_index[i];
      int other_plane  = _other_planes[ _flowdirs[i] ];
      LARCV_DEBUG() << "source plane=" << source_plane
                    << " target plane=" << target_plane
                    << " other plane=" << other_plane
                    << std::endl;
      
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
      _ntrue_pairs[i]  = 0;
      _nfalse_pairs[i] = 0;
      
      // now we can make the src pixel to target pixel choices map
      for ( int ipt=0; ipt<sparsesrc.len(); ipt++ ) {

        // source coordinates
        int   col = (int)sparsesrc.getfeature(ipt,1);
        int   row = (int)sparsesrc.getfeature(ipt,0);
        int   matchable = -1; 

        if ( _has_mctruth )
          matchable = (int)sparsesrc.getfeature(ipt,5+i);

        float umin = _wire_bounds[i][col][0];
        float umax = _wire_bounds[i][col][1];

        std::vector<int> matchable_target_cols;
        
        auto it=targetmap.find(row);
        if ( it!=targetmap.end() ) {
          auto const& target_index_v = targetmap[row];

          std::vector< int > truth_v;
          std::vector< int > within_bounds_target_v;

          // if MC, find the true column match
          float flow = 0;
          int target_col = 0;

          if ( _has_mctruth ) {
            flow = sparsesrc.getfeature(ipt,3+i); // the true flow
            target_col = col + (int)flow;
          }

          int nmatches = 0;
          std::vector<int> tar_col_v;
          for ( size_t idx=0; idx<target_index_v.size(); idx++ ) {
            int tarcol = sptar.getfeature( target_index_v[idx], 1 );

            if ( tarcol<(int)umin || tarcol>(int)umax ) continue;

            // if we enforce 3-plane consistency, we need to check for charge in the other plane
            if ( _use_3plane_constraint ) {
              double y, z;
              larutil::Geometry::GetME()->IntersectionPoint( col, tarcol, (UChar_t)source_plane, (UChar_t)target_plane, y, z );
              Double_t pos[3] = { 0, y, z };
              float other_wire = larutil::Geometry::GetME()->WireCoordinate( pos, other_plane );
              if ( other_wire<0 || (int)other_wire>=(int)larutil::Geometry::GetME()->Nwires(other_plane) )
                continue;
              float other_adc  = ev_adc->Image2DArray()[other_plane].pixel( row, (int)other_wire,__FILE__,__LINE__ );

              bool indead = false;
              if ( other_adc<10.0 ) {

                // not using dead channels, other plane is below threshold. skip this combination
                if ( ev_chstatus==nullptr )
                  continue;

                // if event chstatus pointer is not null, we check the ch status
                int chstatus = ev_chstatus->Status( other_plane ).Status( other_wire );
                if ( chstatus==4 ) {
                  // good channel, so we ignore this match
                  continue;
                }
              }
            }

            // moving on, we store this combination
            
            tar_col_v.push_back( tarcol );
            within_bounds_target_v.push_back( target_index_v[idx] );

	    if ( _has_mctruth && abs(tarcol-target_col)<=_positive_example_distance ) {
	      // within positive example distance
	      if ( !_use_soft_truth || tarcol==target_col ) {
		// if positive example, set to truth vector to 1
		truth_v.push_back(1.0);
	      }
	      else {
		// soft truth
		truth_v.push_back( 2.0/float(tarcol-target_col) ); // arbitrary function
	      }
              _ntrue_pairs[i]++;
              if ( tarcol==target_col) nmatches++;
            }
            else {
              truth_v.push_back(0.0);
              _nfalse_pairs[i]++;              
            }
          }
          
          if ( matchable==1 ) {
            if ( _has_mctruth && nmatches!=1 ) {
              //LARCV_CRITICAL() << "did not find matchable column" << std::endl;
            }

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

          if ( _debug_detailed_output ) {
            LARCV_DEBUG() << "srcpixel[" << ipt << ", (" << row << "," << col << ")] "
                          << " flow (" << flow << ") to targetcol=" << target_col
                          << " matchable=" << matchable
                          << " bounds=[" << umin << "," << umax << "]"
                          << std::endl;
            LARCV_DEBUG() << "  has " << target_index_v.size() << " potential matches " << sstarcol.str()
                          << "  and " << nmatches << " correct match" << std::endl;
          }
          matchmap.add_matchdata( ipt, within_bounds_target_v, truth_v );
        }
        else {
          matchmap.add_matchdata( ipt, std::vector<int>(), std::vector<int>() );
        }
      }//loop over points
      
      _matchdata_v->emplace_back( std::move(matchmap) );
      LARCV_INFO() << "matched map flowdir=" << i << ": matchable hasmatch=" << npix_w_matches << " hasnomatch=" << npix_wo_matches << std::endl;
      
    }//end of loop over flow directions
    
    LARCV_INFO() << "number of true matches:  flow[0]=" << _ntrue_pairs[0]  << "  flow[1]=" << _ntrue_pairs[1]  << std::endl;
    LARCV_INFO() << "number of false matches: flow[0]=" << _nfalse_pairs[0] << "  flow[1]=" << _nfalse_pairs[1] << std::endl;


    LARCV_DEBUG() << "pass sparse images to iomanager" << std::endl;
    std::stringstream sparseout_name;
    sparseout_name << "larflow_plane" << _source_plane;
    larcv::EventSparseImage* ev_out = (larcv::EventSparseImage*)mgr.get_data(larcv::kProductSparseImage, sparseout_name.str() );
    ev_out->Emplace( std::move( spimg_v ) );
    
    if ( _use_ana_tree ) {
      LARCV_DEBUG() << "save flow match maps to ana tree" << std::endl;    
      _ana_tree->Fill();
    }

    LARCV_DEBUG() << "done" << std::endl;    
    return true;
  }

  void PrepFlowMatchData::finalize() {
    if ( _use_ana_tree )
      _ana_tree->Write();
  }

  void PrepFlowMatchData::_setup_ana_tree() {
    if ( _use_ana_tree ) 
      LARCV_DEBUG() << "create tree, make container, set branch" << std::endl;
    else
      LARCV_DEBUG() << "not storing data in ana tree" << std::endl;

    _matchdata_v = new std::vector< FlowMatchMap >();

    if ( !_use_ana_tree )
      return;


    char treename[50];
    sprintf(treename,"flowmatchdata_plane%d",_source_plane);
    
    _ana_tree = new TTree(treename,"Provides map from source to target pixels to match");
    _ana_tree->Branch( "matchmap",    _matchdata_v );
    _ana_tree->Branch( "nfalsepairs", _nfalse_pairs, "nfalsepairs[2]/I" );
    _ana_tree->Branch( "ntruepairs",  _ntrue_pairs,  "ntruepairs[2]/I" );
    
  }

  /**
   * internal: define ranges of wires on the target planes that overlap with the source plane.
   *
   * called by initialize. also setups key variables.
   *
   */
  void PrepFlowMatchData::_extract_wire_overlap_bounds() {

    const larutil::Geometry* geo = larutil::Geometry::GetME();

    // set the flow directions we will evaluate

    std::string src_plane_name;    
    switch ( _source_plane ) {
    case 2:
      _flowdirs[0] = kY2U;
      _flowdirs[1] = kY2V;
      src_plane_name = "Y";
      break;
    case 0:
      _flowdirs[0] = kU2V;
      _flowdirs[1] = kU2Y;
      src_plane_name = "U";      
      break;
    case 1:
      _flowdirs[0] = kV2U;
      _flowdirs[1] = kV2Y;
      src_plane_name = "V";
      break;
    default:
      throw std::runtime_error("PrepFlowMatchData::_extract_wire_overlap_bounds: bad source plane");
      break;
    }

    // loop over flow directions
    for (int i=0; i<2; i++ ) {
      
      int target_plane = _target_planes[_flowdirs[i]];
      
      LARCV_DEBUG() << "defined overlapping wire bounds for " << src_plane_name << "[" << _source_plane << "]"
                    << "->plane[" << target_plane << "]" << std::endl;
      
      _wire_bounds[i].clear();


      for (int isrc=0; isrc<(int)geo->Nwires(_source_plane); isrc++ ) {
        Double_t xyzstart[3];
        Double_t xyzend[3];
        geo->WireEndPoints( (UChar_t)_source_plane, (UInt_t)isrc, xyzstart, xyzend );

        float u1 = geo->WireCoordinate( xyzstart, (UInt_t)target_plane );
        float u2 = geo->WireCoordinate( xyzend,   (UInt_t)target_plane );

        float umin = (u1<u2) ? u1 : u2;
        float umax = (u1>u2) ? u1 : u2;

        umin -= 5.0;
        umax += 5.0;

        if ( umin<0 ) umin = 0;
        if ( umax<0 ) umax = 0;

        if ( (int)umin>=geo->Nwires(target_plane) ) umin = (float)geo->Nwires(target_plane)-1;
        if ( (int)umax>=geo->Nwires(target_plane) ) umax = (float)geo->Nwires(target_plane)-1;

        _wire_bounds[i][isrc] = std::vector<int>{ (int)umin, (int)umax };
        //LARCV_DEBUG() << " src[" << isrc << "] -> plane[" << i << "]: (" << umin << "," << umax << ")" << std::endl;
      }
    }
    LARCV_DEBUG() << "defined overlapping wire bounds" << std::endl;
    //std::cin.get();
  }
  
}
