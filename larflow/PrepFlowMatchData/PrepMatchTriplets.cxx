#include "PrepMatchTriplets.h"
#include "FlowTriples.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

#include <sstream>
#include <ctime>
#include <algorithm>    // std::shuffle
#include <random>       // std::default_random_engine
#include <chrono>       // std::chrono::system_clock

#include "larlite/LArUtil/LArProperties.h"
#include "larlite/LArUtil/DetectorProperties.h"
#include "larlite/LArUtil/Geometry.h"
#include "larcv/core/PyUtil/PyUtils.h"
#include "larflow/LArFlowConstants/LArFlowConstants.h"
#include "ublarcvapp/UBWireTool/UBWireTool.h"
#include "ublarcvapp/UBImageMod/EmptyChannelAlgo.h"
#include "ublarcvapp/RecoTools/DetUtils.h"

#include "TRandom3.h"

namespace larflow {
namespace prep {

  bool PrepMatchTriplets::_setup_numpy = false;
  
  /**
   * clear the containers we fill when we run process
   *
   */
  void PrepMatchTriplets::clear()
  {
    // clear the output data
    for ( auto& matchdata : _match_triplet_v )
      matchdata.clear();

    // clear meta data for an event
    _shower_daughter2mother.clear();
    _instance2class_map.clear();
  }
  
  /**
   * @brief convenience function that gets the data needed from an larcv::IOManager instance and runs the process method.
   *
   * @param[in] iolcv   larcv::IOManager instance containing needed data objects
   * @param[in] wire_producer     Name of tree containing event ADC images
   * @param[in] chstatus_producer Name of tree containing badch status
   * @param[in] adc_threshold     Threshold value for pixels we will consider (typically 10.0)
   * @param[in] check_wire_intersection Check that triplet produces a good wire intersection inside the TPC.
   *                                    Also saves the 3D position for each triplet. (Makes slower; For debug.)
   */
  void PrepMatchTriplets::process( larcv::IOManager& iolcv,
                                   std::string wire_producer,
                                   std::string chstatus_producer,
                                   const float adc_threshold,
                                   const bool check_wire_intersection )
  {

    clear();
    _match_triplet_v.clear();

    // get images
    larcv::EventImage2D* ev_adc
      = (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D,wire_producer);
    auto const& adc_v = ev_adc->as_vector();
    
    // get chstatus
    larcv::EventChStatus* ev_badch
      = (larcv::EventChStatus*)iolcv.get_data(larcv::kProductChStatus,chstatus_producer);

    std::vector< larcv::Image2D > badch_v;
    
    if ( ev_badch->ChStatusMap().size()==0 ) {
      // leave blank
      LARCV_NORMAL() << "No bad channel info so making blanks" << std::endl;
      // for ( auto const& adc : ev_adc->as_vector() ) {
      // 	larcv::Image2D badch_blank(adc.meta());
      // 	badch_blank.paint(0.0);
      // 	badch_v.emplace_back( badch_blank );
      // }
    }
    else {
      // create bad channel maker algo
      LARCV_NORMAL() << "Have bad channel info make badchannel images" << std::endl;
      ublarcvapp::EmptyChannelAlgo badchmaker;
      badch_v = badchmaker.makeGapChannelImage( ev_adc->as_vector(), *ev_badch,
						4, 3, 2400, 6*1008,
						3456, 6, 1,
						1.0, 100, -1.0 );
    }

    // to do: create bad channel maker algo
    // ublarcvapp::EmptyChannelAlgo badchmaker;
    
    // we loop over tpcs
    auto const geom = larlite::larutil::Geometry::GetME();

    for ( int icryo=0; icryo<(int)geom->Ncryostats(); icryo++ ) {
      for (int itpc=0; itpc<geom->NTPCs(icryo); itpc++) {	
	std::vector< const larcv::Image2D*  > tpc_adc_v;
	std::vector< const larcv::Image2D*  > tpc_badch_v;

	int nplanes = geom->Nplanes( itpc, icryo );
	for (int p=0; p<nplanes; p++) {

	  int planeindex = geom->GetSimplePlaneIndexFromCTP( icryo, itpc, p );
	  for ( int iplane=0; iplane<(int)adc_v.size(); iplane++ ) {
	    auto& img = adc_v[iplane];
	    if ( img.meta().id()==planeindex ) {
	      tpc_adc_v.push_back( &adc_v.at(iplane) );
	      tpc_badch_v.push_back( &badch_v.at(iplane) );
	      LARCV_NORMAL() << "Store adc_v image[" << iplane << "] for (plane,tpc,cryo)=(" << p << "," << itpc << "," << icryo << ")" << std::endl;
	    }	      
	  }
	}
	
	process_tpc_v2( tpc_adc_v, tpc_badch_v, adc_threshold, itpc, icryo );
	
      }
    }
    
  } 
  
  /**
   * @brief make the possible hit triplets from all the wire plane images (version 2)
   *
   * this function is expected to populate the following data members:
   *  _sparseimg_vv
   *  _imgmeta_v
   *  _triplet_v
   *  _flowdir_v
   * it also uses different approach for new larmatch
   *
   * @param[in] adc_v   Vector of wire plane images.
   * @param[in] badch_v Vector of bad channel images.
   * @param[in] adc_threshold Threshold value for pixels we will consider.
   * @param[in] tpcid   The TPC index within the cryostat
   * @param[in] cryoid  The cryostat index
   */
  void PrepMatchTriplets::process_tpc_v2( const std::vector<const larcv::Image2D*>& adc_v,
					  const std::vector<const larcv::Image2D*>& badch_v,
					  const float adc_threshold,
					  const int tpcid, const int cryoid )
  {
    
    std::clock_t start = std::clock();

    // The data object we're filling
    MatchTriplets matchdata;
    matchdata._tpcid  = tpcid;
    matchdata._cryoid = cryoid;
      
    // do we have any data?
    int npix = 0;
    for ( auto const& padc : adc_v ) {
      for ( auto const& pix : padc->as_vector() ) {
	if (pix>=adc_threshold)
	  npix++;

	if ( npix>=10 )
	  break;
      }
      break;
    }

    // empty TPC, return
    if (npix<10) {
      for ( auto const& padc : adc_v ) {
	matchdata._imgmeta_v.push_back( padc->meta() );
	std::vector< FlowTriples::PixData_t > sparseimg_blank;
	matchdata._sparseimg_vv.push_back( sparseimg_blank );
      }
      LARCV_INFO() << "Fill Blanks for this TPC" << std::endl;
      _match_triplet_v.emplace_back( std::move(matchdata) );
      return;
    }

    // do we have bad channel info?
    bool have_badch = (badch_v.size()>0) ? true : false;
    
    // first we make a common sparse image
    std::vector< std::vector<FlowTriples::PixData_t> > sparseimg_vv =
      larflow::prep::FlowTriples::make_initial_sparse_image( adc_v, adc_threshold );

    // we index the pixels for each plane
    std::vector< std::map< std::pair<int,int>, int > > map_sparseimg_pix2index_v( sparseimg_vv.size() );
    
    for ( size_t p=0; p<sparseimg_vv.size(); p++) {
      auto const& pix_v = sparseimg_vv[p];
      std::map< std::pair<int,int>, int >& pix2index = map_sparseimg_pix2index_v[p];
      pix2index.clear();
      for (size_t ipix=0; ipix<pix_v.size(); ipix++) {
	pix2index[ std::pair<int,int>( pix_v[ipix].row, pix_v[ipix].col ) ] = (int)ipix;
	//std::cout << "plane[" << p << "] (" << pix_v[ipix].row << "," << pix_v[ipix].col << ")" << std::endl;
      }
    }
    
    for ( size_t p=0; p<adc_v.size(); p++ ) {
      auto const& padc = adc_v[p];
      matchdata._imgmeta_v.push_back( padc->meta() );
      matchdata._sparseimg_vv.emplace_back( std::move(sparseimg_vv.at(p)) );
    }
      
    // Load the overlap matrices if not yet done
    _load_overlap_matrices();

    int start_plane = 0;
    int nplanes = adc_v.size();
    auto const& start_img = *adc_v.at(start_plane);
    auto const& meta = start_img.meta();
    int nrows = meta.rows();
    int ncols = meta.cols();
    std::vector< std::vector<int> > plane_combos_v;
    plane_combos_v.push_back( std::vector<int>( {0,1,2} ) );
    plane_combos_v.push_back( std::vector<int>( {1,2,0} ) );
    plane_combos_v.push_back( std::vector<int>( {0,2,1} ) );    

    auto const geom    = larlite::larutil::Geometry::GetME();
    auto const detprop = larutil::DetectorProperties::GetME();    

    std::vector< std::vector<const TMatrixD*> > intersect_vv;

    for (int ii=0; ii<(int)plane_combos_v.size(); ii++) {
      int ipl1 = plane_combos_v[ii][0];
      int ipl2 = plane_combos_v[ii][1];
      int ipl3 = plane_combos_v[ii][2];
      std::vector<int> plane_index1 = { cryoid, tpcid, ipl1, ipl2, ipl3 };
      auto it1 = _m_planeid_to_tree_entry.find( plane_index1 );
      if ( it1==_m_planeid_to_tree_entry.end() )
	LARCV_CRITICAL() << "bad plane index1=" << it1->second << std::endl;
      const TMatrixD& intersect1 = _matrix_list_v.at( it1->second );
      
      std::vector<const TMatrixD*> intersect_v;
      intersect_v.push_back( &intersect1 );
      intersect_vv.push_back( intersect_v );
    }

    size_t ntriplets_made = 0;
    std::map< std::vector<int>,int > triplet_map;
    size_t nrepeated = 0;
    
    for (int irow=0; irow<nrows; irow++) {
      float tick = meta.pos_y(irow);      
      // loop through the plane colums and get columns above thresh
      //std::cout << "irow[" << irow << "] -----------------------" << std::endl;
      std::vector< std::vector<int> > plane_cols(nplanes);
      for (int iplane=0; iplane<nplanes; iplane++) {
	plane_cols[iplane].clear();
     	int splaneid = start_plane+iplane;
     	auto const& pimg = adc_v.at(splaneid);
	//std::cout << "  plane[" << iplane << " ]: ";
     	for (int icol=0; icol<(int)pimg->meta().cols(); icol++) {
     	  if ( pimg->pixel(irow,icol)>=adc_threshold ) {
     	    plane_cols[iplane].push_back(icol);
	    //std::cout << " " << icol;
	  }
     	}
	//std::cout << " :: ncols=" << plane_cols[iplane].size() << std::endl;
      }
      
      for (int ii=0; ii<(int)plane_combos_v.size(); ii++) {
	int ipl1 = plane_combos_v[ii][0];
     	int ipl2 = plane_combos_v[ii][1];
	int ipl3 = plane_combos_v[ii][2];
     	const TMatrixD& intersect = *(intersect_vv[ii][0]);
     	//std::cout << ii << ": " << ipl1 << "," << ipl2 << std::endl;
	//std::cout << "matrix: " << intersect.GetNrows() << "x" << intersect.GetNcols() << " elemns=" << intersect.GetNoElements() << std::endl;
 	for (auto idx1 : plane_cols[ipl1] ) {
	  for (auto idx2: plane_cols[ipl2] ) {
	    //std::cout << "(" << idx1 << "," << idx2 << ")" << std::endl;
	    if ( intersect[idx1][idx2]> 0 ) {	    
	      // register doublet
	      //std::cout << " register doublet: " << idx1 << "," << idx2 << std::endl;

	      // other plane wire
	      int otherwire = intersect[idx1][idx2]-1;
	      float pixvalue = adc_v[start_plane+ipl3]->pixel(irow,otherwire);

	      // one condition of accepting spacepoint proposal
	      bool abovethresh = pixvalue>=adc_threshold;

	      // handle dead channels (microboone)
	      bool deadch_pass = false;
	      if ( have_badch ) {
		float badchval = badch_v[start_plane+ipl3]->pixel(irow,otherwire);
		if ( badchval>0 )
		  deadch_pass = true;
	      }

	      // missing induction plane
	      bool induction_pass = false;
	      if ( ipl3==0 && _kAllowInductionPass ) { //fabs(pixvalue)>2.0 ) {
		induction_pass = true;
	      }
	      
	      if ( abovethresh || deadch_pass || induction_pass  ) {
		// * to do: how to forgive induction plane
		
		std::vector<int> wirecoord(4,0);
		wirecoord[ipl1] = idx1;
		wirecoord[ipl2] = idx2;
		wirecoord[ipl3] = otherwire;
		wirecoord[3]    = (int)tick;

		// we need to insert the dead or induction pixel into the sparsemap
		if ( induction_pass || deadch_pass ) {
		  auto& pl3map = map_sparseimg_pix2index_v[ipl3];
		  auto it_img = pl3map.find( std::pair<int,int>(irow,otherwire) );
		  if ( it_img==pl3map.end() ) {
		    auto& pl3sparseimg = matchdata._sparseimg_vv.at(ipl3);
		    int newindex = pl3sparseimg.size();
		    pl3sparseimg.push_back( FlowTriples::PixData_t( irow, otherwire, fabs(pixvalue), newindex ) );
		    pl3map[ std::pair<int,int>(irow,otherwire) ] = newindex;
		    // std::cout << "  IND pass CTP=(" << cryoid << "," << tpcid << "," << ipl3 << "): "
		    // 	      << " row=" << irow << " tick=" << (int)tick
		    // 	      << " wires=(" << wirecoord[0] << "," << wirecoord[1] << "," << wirecoord[2] << ") "
		    // 	      << "adding pix(row,col)=(" << irow << "," << wirecoord[2] << ") "
		    // 	      << "pixval=" << pixvalue
		    // 	      << std::endl;
		  }
		}
		else if ( abovethresh ) {
		  // std::cout << "  3-PLANE CTP=(" << cryoid << "," << tpcid << "," << ipl3 << "): "
		  // 	    << " row=" << irow << " tick=" << (int)tick
		  // 	    << " wires=(" << wirecoord[0] << "," << wirecoord[1] << "," << wirecoord[2] << ") "
		  // 	    << "adding pix(row,col)=(" << irow << "," << wirecoord[2] << ") "
		  // 	    << "pixval=" << pixvalue
		  // 	    << std::endl;
		}
		
		// get triplet index
		std::vector<int> triplet(4,0);
		for (size_t p=0; p<3; p++) {
		  auto it_img = map_sparseimg_pix2index_v[p].find( std::pair<int,int>(irow,wirecoord[p]) );
		  if ( it_img==map_sparseimg_pix2index_v[p].end() ) {
		    LARCV_CRITICAL() << "Could not find pixel in sparse-image[" << p << "]."
				     << "pix (row,col)=(" << irow << "," << wirecoord[p] << ")"
				     << std::endl;
		    throw std::runtime_error("PrapMatchTriplets has issue mapping index to sparse pixel image");
		  }
		  triplet[p] = it_img->second;
		}
		triplet[3] = irow;

		auto it_map = triplet_map.find( triplet );
		if ( it_map==triplet_map.end() ) {
		  // new triplet
		  matchdata._triplet_v.push_back( triplet );
		  matchdata._wirecoord_v.push_back( wirecoord );
		
		  TVector3 pos(0,0,0);
		  std::vector<float> fpos(3,0);
		  int ch1 = geom->PlaneWireToChannel( idx1, ipl1, tpcid, cryoid );
		  int ch2 = geom->PlaneWireToChannel( idx2, ipl2, tpcid, cryoid );
		  bool crosses = geom->ChannelsIntersect( ch1, ch2, pos );
		  for (int i=0; i<3; i++)
		    fpos[i] = pos[i];
		  fpos[0] = detprop->ConvertTicksToX(tick,0,tpcid,cryoid);
		  matchdata._pos_v.push_back( fpos );
		  
		  std::vector<int> trip_cryo_tpc = { cryoid, tpcid };
		  matchdata._trip_cryo_tpc_v.push_back( trip_cryo_tpc );

		  matchdata._flowdir_v.push_back( larflow::LArFlowConstants::getFlowDirection( ipl1, ipl2  ) );
		  
		  ntriplets_made++;
		  triplet_map[triplet] = 1;
		}
		else {
		  nrepeated++;
		}
		
	      }// if otherwire has charge
	    }//if intersects
	  }// loop over plane cols (2)
	}// loop over plane cols (1)
	
	//std::cout << "trips registered this loop: " << trips_registered << std::endl;
	//std::cout << "tot triplets: " << triplet_v.size() << std::endl;
      }//end of plane combo loop
	
      //std::cout << "[enter] to continue." << std::endl;
      //std::cin.get();
    }// end of row loop

    std::clock_t end = std::clock();
    LARCV_NORMAL() << "made total of " << matchdata._triplet_v.size() << " nrepeated=" << nrepeated
		   << " unique index triplets. time elapsed=" << float(end-start)/float(CLOCKS_PER_SEC)
		   << std::endl;
    // LARCV_NORMAL() << "number removed for not intersecting: " << n_not_crosses << std::endl;
    // LARCV_NORMAL() << " num zero triplets: " << n_zero_triplets << std::endl;

    _match_triplet_v.emplace_back( std::move(matchdata) );

  }//end of process method
  

  /**
   * @brief load the wire overlap matrices
   */
  void PrepMatchTriplets::_load_overlap_matrices( bool force_reload )
  {
    if ( !force_reload && _overlap_matrices_loaded )
      return; // alraedy loaded
    
    // LOAD THE INTERSECTION DATA
    TFile fmatrices( _input_overlap_filepath.c_str() );
    TTree* intersectiondata = (TTree*)fmatrices.Get("intersectdata");
    int cryostat = 0;
    int tpc = 0;
    std::vector<int>* p_plane_indices = 0;    
    int dim1 = 0;
    int dim2 = 0;
    std::vector< TMatrixD >* p_matrix_v = 0;
    intersectiondata->SetBranchAddress( "cryostat", &cryostat );
    intersectiondata->SetBranchAddress( "tpc",      &tpc );
    intersectiondata->SetBranchAddress( "dim1",     &dim1 );
    intersectiondata->SetBranchAddress( "dim2",     &dim2 );  
    intersectiondata->SetBranchAddress( "plane_indices",  &p_plane_indices );
    intersectiondata->SetBranchAddress( "matrix_v", &p_matrix_v );

    _matrix_list_v.clear();
    _m_planeid_to_tree_entry.clear();

    int nentries = intersectiondata->GetEntries();
    for (int i=0; i<nentries; i++) {
      intersectiondata->GetEntry(i);
      std::vector<int> index_v = { cryostat, tpc, p_plane_indices->at(0), p_plane_indices->at(1), p_plane_indices->at(2) };
      LARCV_INFO() << "matrix[" << i << "] cryo=" << cryostat << " tpc=" << tpc << " p1=" << index_v[2] << " p2=" << index_v[3] << std::endl;
      TMatrixD mat(p_matrix_v->at(0));    
      _matrix_list_v.emplace_back( std::move(mat) );
      _m_planeid_to_tree_entry[ index_v ] = (int)_matrix_list_v.size()-1;
    }

    LARCV_INFO() << "LOADED MATRICES" << std::endl;
    _overlap_matrices_loaded = true;
  }

  /**
   * @brief make all the initial truth labels
   *
   * runs all the truth label making functions. 
   *
   * @param[in] iolcv larcv IO manager containing event data
   */
  void PrepMatchTriplets::process_truth_labels( larcv::IOManager& iolcv,
                                                larlite::storage_manager& ioll,
                                                std::string wire_producer )
  {
    
    larcv::EventImage2D* ev_adc
      = (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D,wire_producer);
    larcv::EventImage2D* ev_larflow =
      (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D,"larflow");
    larcv::EventImage2D* ev_instance =
      (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D,"instance");
    larcv::EventImage2D* ev_ancestor =
      (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D,"ancestor");
    larcv::EventImage2D* ev_segment =
      (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D,"segment");


    larlite::event_mctrack* ev_mctrack
      = (larlite::event_mctrack*)ioll.get_data(larlite::data::kMCTrack, "mcreco" );
    larlite::event_mcshower* ev_mcshower
      = (larlite::event_mcshower*)ioll.get_data(larlite::data::kMCShower, "mcreco" );
    fill_daughter2mother_map( *ev_mcshower );
    fill_class_map( *ev_mctrack, *ev_mcshower );
    
    make_truth_vector( ev_larflow->as_vector(), ev_instance->as_vector() );
    make_instanceid_vector( ev_instance->as_vector() );
    make_ancestorid_vector( ev_ancestor->as_vector() );
    make_segmentid_vector( ev_segment->as_vector(), ev_adc->as_vector() ); // segment image only labels neutrino pixels
    make_origin_vector_frommcreco( ioll );
  }


  /**
   * @brief use larflow truth images to assign good versus bad match for triplets
   *
   * this method populates the values for:
   * _truth_v: 1 or 0 if triplet is a correct match
   * _truth_2plane_v: value for all 6 flows, 1 or 0 if correct flow
   *
   * There should be a value of either (1) for correct triplet
   *  and (2) for false triplet
   *
   * @param[in] larflow_v Vector of Image2D which contain true flow information between planes.
   * @param[in] larflow_v Vector of Image2D which contain a Geant4 trackID in each pixel
   *
   */
  void PrepMatchTriplets::make_truth_vector( const std::vector<larcv::Image2D>& larflow_v,
					     const std::vector<larcv::Image2D>& instance_v )
  {

    for ( auto& matchdata : _match_triplet_v ) {

      std::vector< const larcv::Image2D* > plarflow_v
	= ublarcvapp::recotools::DetUtils::getTPCImages( larflow_v, matchdata._tpcid, matchdata._cryoid );
      
      std::vector< const larcv::Image2D* > pinstance_v
	= ublarcvapp::recotools::DetUtils::getTPCImages( instance_v, matchdata._tpcid, matchdata._cryoid );
      
      LARCV_NORMAL() << " (cryo,tpcid)=" << matchdata._cryoid << "," << matchdata._tpcid << ") num larflow truth images=" << plarflow_v.size() << std::endl;
      
      if ( plarflow_v.size()!=6 ) {
	continue;
      }

      int ntriplets = matchdata._triplet_v.size();
      
      matchdata._truth_v.clear();
      //matchdata._truth_2plane_v.clear();
      matchdata._truth_v.resize( ntriplets, 0 );
      //matchdata._truth_2plane_v.resize( ntriplets );
      matchdata._match_maxspan_v.clear();
      matchdata._match_maxspan_v.resize( ntriplets, 0 );    
      matchdata._match_minspan_v.clear();
      matchdata._match_minspan_v.resize( ntriplets, 0 );    
      matchdata._match_cyclespan_v.clear();
      matchdata._match_cyclespan_v.resize( ntriplets, 0 );    

      const int true_match_span = 3;
      const int min_required_connections = 1;

      int ntriplet_truth = 0;
      std::vector< int > ndoublet_truth( (int)larflow::kNumFlows, 0 );
    
      for ( size_t itrip=0; itrip<matchdata._triplet_v.size(); itrip++ ) {
	// for each triplet, we look for truth flows that connect the planes
	auto const& triplet = matchdata._triplet_v[itrip];
	larflow::FlowDir_t flow_dir_origin = matchdata._flowdir_v[itrip];
	//matchdata._truth_2plane_v[itrip].resize( (int)larflow::kNumFlows, 0 );
      
	// for debug
	//if ( flow_dir_origin!=kY2V ) continue;

	int srcplane, tarplane;
	larflow::LArFlowConstants::getFlowPlanes( flow_dir_origin, srcplane, tarplane );
	int othplane = larflow::LArFlowConstants::getOtherPlane( srcplane, tarplane );
      
	std::vector< const FlowTriples::PixData_t* > pix_v( matchdata._sparseimg_vv.size() );
	pix_v[srcplane] = &matchdata._sparseimg_vv[srcplane][ triplet[srcplane] ];
	pix_v[tarplane] = &matchdata._sparseimg_vv[tarplane][ triplet[tarplane] ];
	pix_v[othplane] = &matchdata._sparseimg_vv[othplane][ triplet[othplane] ];

	int on_instance_src = 0;
	if ( pinstance_v[srcplane]->pixel( pix_v[srcplane]->row, pix_v[srcplane]->col )>-1 )
	  on_instance_src = 1;
	int on_instance_tar = 0;
	if ( pinstance_v[tarplane]->pixel( pix_v[tarplane]->row, pix_v[tarplane]->col )>-1 )
	  on_instance_tar = 1;
	int on_instance_oth = 0;
	if ( pinstance_v[othplane]->pixel( pix_v[othplane]->row, pix_v[othplane]->col )>-1 )
	  on_instance_oth = 1;

	int ngood_connections = 0;
	std::vector<int> match_span(6);

	// [0] we follow the original flow direction: src->target planes --------------------	
	float pixflow0  = plarflow_v[(int)flow_dir_origin]->pixel( pix_v[srcplane]->row, pix_v[srcplane]->col );
	int target_col0 = pix_v[srcplane]->col + (int)pixflow0;
	match_span[0] = abs(target_col0-pix_v[tarplane]->col);
	
	if ( (fabs(pixflow0)!=0 || (int(pixflow0)==0 && on_instance_src==1)) // checks
	     && match_span[0]<true_match_span ) {
	  ngood_connections++;
	  //matchdata._truth_2plane_v[itrip][(int)flow_dir_origin] = 1;
	  ndoublet_truth[(int)flow_dir_origin]++;
	}

	// [1] other+target swap flow direction: src->other --------------------
	int flowdir1 = (int)larflow::LArFlowConstants::getFlowDirection( srcplane, othplane );
	float pixflow1  = plarflow_v[(int)flowdir1]->pixel( pix_v[srcplane]->row, pix_v[srcplane]->col );
	int target_col1 = pix_v[srcplane]->col + (int)pixflow1;
	match_span[1] = abs(target_col1-pix_v[othplane]->col);
	
	if ( (fabs(pixflow1)!=0 || (int(pixflow1)==0 && on_instance_src==1)) // checks
	     && match_span[1]<true_match_span ) {
	  ngood_connections++;
	  //matchdata._truth_2plane_v[itrip][(int)flow_dir_origin] = 1;
	  ndoublet_truth[(int)flowdir1]++;
	}

	// [2] src->(target->other) flow direction --------------------
	int flowdir2 = (int)larflow::LArFlowConstants::getFlowDirection( tarplane, othplane );
	float pixflow2  = plarflow_v[(int)flowdir2]->pixel( pix_v[tarplane]->row, pix_v[tarplane]->col );
	int target_col2 = pix_v[tarplane]->col + (int)pixflow2;
	match_span[2] = abs(target_col2-pix_v[othplane]->col);
	
	if ( (fabs(pixflow2)!=0 || (int(pixflow2)==0 && on_instance_tar==1)) // checks
	     && match_span[2]<true_match_span ) {
	  ngood_connections++;
	  //matchdata._truth_2plane_v[itrip][(int)flow_dir_origin] = 1;
	  ndoublet_truth[(int)flowdir2]++;
	}

	// [3] src->(other->target) flow direction --------------------
	int flowdir3 = (int)larflow::LArFlowConstants::getFlowDirection( othplane, tarplane );
	float pixflow3  = plarflow_v[(int)flowdir3]->pixel( pix_v[othplane]->row, pix_v[othplane]->col );
	int target_col3 = pix_v[othplane]->col + (int)pixflow3;
	match_span[3] = abs(target_col3-pix_v[tarplane]->col);
	
	if ( (fabs(pixflow3)!=0 || (int(pixflow3)==0 && on_instance_oth==1)) // checks
	     && match_span[3]<true_match_span ) {
	  ngood_connections++;
	  //matchdata._truth_2plane_v[itrip][(int)flow_dir_origin] = 1;
	  ndoublet_truth[(int)flowdir3]++;
	}

	// [4] other->src flow direction --------------------
	int flowdir4 = (int)larflow::LArFlowConstants::getFlowDirection( othplane, srcplane );
	float pixflow4  = plarflow_v[(int)flowdir4]->pixel( pix_v[othplane]->row, pix_v[othplane]->col );
	int target_col4 = pix_v[othplane]->col + (int)pixflow4;
	match_span[4] = abs(target_col4-pix_v[srcplane]->col);
	
	if ( (fabs(pixflow4)!=0 || (int(pixflow4)==0 && on_instance_oth==1)) // checks
	     && match_span[4]<true_match_span ) {
	  ngood_connections++;
	  ndoublet_truth[(int)flowdir4]++;
	}	

	// [5] target->src flow direction --------------------
	int flowdir5 = (int)larflow::LArFlowConstants::getFlowDirection( tarplane, srcplane );
	float pixflow5  = plarflow_v[(int)flowdir5]->pixel( pix_v[tarplane]->row, pix_v[tarplane]->col );
	int target_col5 = pix_v[tarplane]->col + (int)pixflow5;
	match_span[5] = abs(target_col5-pix_v[srcplane]->col);
	
	if ( (fabs(pixflow5)!=0 || (int(pixflow5)==0 && on_instance_tar==1)) // checks
	     && match_span[5]<true_match_span ) {
	  ngood_connections++;
	  ndoublet_truth[(int)flowdir5]++;
	}	

	int max_span = 0;
	int min_span = 2000;
	for (auto const& span : match_span ) {
	  if ( span>max_span )
	    max_span = span;
	  if ( span<min_span )
	    min_span = span;
	}
	int cycle_span = match_span[0]+match_span[2]+match_span[4];
	
	matchdata._match_maxspan_v[itrip]   = max_span;
	matchdata._match_minspan_v[itrip]   = max_span;
	matchdata._match_cyclespan_v[itrip] = cycle_span;
	if ( ngood_connections>=min_required_connections ) {
	  matchdata._truth_v[itrip] = 1;
	  ntriplet_truth++;
	}
	else {
	  matchdata._truth_v[itrip] = 0;
	}
      }//end of trips loop

      LARCV_NORMAL() << "=== (cryoid,tpcid)=(" << matchdata._cryoid << "," << matchdata._tpcid << ") =======" << std::endl;
      LARCV_NORMAL() << " number of triplets: " << matchdata._triplet_v.size() << std::endl;
      LARCV_NORMAL() << "  number of sparse pixels: [ "
		     << matchdata._sparseimg_vv[0].size() << ", "
		     << matchdata._sparseimg_vv[1].size() << ", "
		     << matchdata._sparseimg_vv[2].size() << " ]"
		     << std::endl;

      // clear to save disk space.
      // we'll start with using cyclespan
      matchdata._match_minspan_v.clear();
      matchdata._match_maxspan_v.clear();

      LARCV_NORMAL() << "  number of true-match triplets: " << ntriplet_truth << std::endl;
      std::stringstream ss;
      ss << "  doublet truth: [";
      for (auto& n : ndoublet_truth ) ss << " " << n << ",";
      ss << " ]";
      LARCV_NORMAL() << ss.str() << std::endl;
    }//end of loop over matchtriplet data (from different tpcs)

    
  }

  /**
   * @brief stores map between daughter geant4 track ids to mother ids
   *
   * Used in conjuction with the 2D instance image labels.
   *
   */
  void PrepMatchTriplets::fill_daughter2mother_map( const std::vector<larlite::mcshower>& shower_v )
  {

    _shower_daughter2mother.clear();
    
    for (auto const& shower : shower_v ) {
      long showerid = shower.TrackID();
      if ( showerid<0 ) showerid *= -1;
      const std::vector<unsigned int>& dlist = shower.DaughterTrackID();      
      for (auto const& daughterid : dlist ) {
        _shower_daughter2mother[(unsigned long)daughterid]= (unsigned long)showerid;
      }
    }
    
  }

  /**
   * @brief stores map between instance id and particle class
   *
   * Used in conjuction with the SSNet class labels
   *
   */
  void PrepMatchTriplets::fill_class_map( const std::vector<larlite::mctrack>&  track_v,
                                          const std::vector<larlite::mcshower>& shower_v )
  {

    _instance2class_map.clear();
    
    for (auto const& shower : shower_v ) {
      long showerid = shower.TrackID();
      if ( showerid<0 ) showerid *= -1;
      int  pid = shower.PdgCode();
      _instance2class_map[(unsigned long)showerid] = pid;
      for ( auto const& daughterid : shower.DaughterTrackID() ) {
        long id = (daughterid<0) ? daughterid*-1 : daughterid;
        _instance2class_map[(unsigned long)id] = pid;
      }
    }
    
    for (auto const& track : track_v ) {
      long trackid = track.TrackID();
      if ( trackid<0 ) trackid *= -1;
      int  pid = track.PdgCode();
      _instance2class_map[(unsigned long)trackid] = pid;
    }

    for (auto it=_instance2class_map.begin(); it!=_instance2class_map.end(); it++ ) {
      int larcv_class = 0;
      switch ( it->second ) {
      case 11:
      case -11:
        larcv_class = (int)larcv::kROIEminus;
        break;
      case 13:
      case -13:
        larcv_class = (int)larcv::kROIMuminus;
        break;
      case 211:
      case -211:
        larcv_class = (int)larcv::kROIPiminus;
        break;
      case 2212:
      case 2112:
        larcv_class = (int)larcv::kROIProton;
        break;
      case 22:
        larcv_class = (int)larcv::kROIGamma;
        break;
      case 111:
        larcv_class = (int)larcv::kROIPizero;
        break;
      case 130:
      case 310:
      case 311:
      case 312:
        larcv_class = (int)larcv::kROIKminus;
        break;
      default:
        larcv_class = (int)larcv::kROIUnknown;
        break;
      }
      it->second = larcv_class;
    }//end of iterator loop
    
  }

  /**
   * @brief use instance ID image to label space points
   *
   * @param[in] instance_img_v Vector of Image2D which contain track id.
   *
   */
  void PrepMatchTriplets::make_instanceid_vector( const std::vector<larcv::Image2D>& instance_img_v )
  {

    for ( auto& matchdata : _match_triplet_v ) {
      
      matchdata._instance_id_v.resize( matchdata._triplet_v.size(), 0);

      std::vector< const larcv::Image2D* > pinstance_v
	= ublarcvapp::recotools::DetUtils::getTPCImages( instance_img_v, matchdata._tpcid, matchdata._cryoid );
      
      int nids = 0;
      
      for ( size_t itrip=0; itrip<matchdata._triplet_v.size(); itrip++ ) {

	if ( matchdata._truth_v[itrip]==0 ) {
	  matchdata._instance_id_v[itrip] = 0;
	  continue;
	}
      
	// for each triplet, we look for truth flows that connect the planes
	auto const& triplet = matchdata._triplet_v[itrip];
	std::vector<int> imgcoord = {0,0,0,triplet[3]};
	for (int p=0; p<3; p++ ) {
	  imgcoord[p] = matchdata._sparseimg_vv[p][triplet[p]].col;
	}
	
	std::map< int, int > id_votes;

	for (int p=0; p<3; p++ ) {
	  long plane_id = pinstance_v[p]->pixel( imgcoord[3], imgcoord[p], __FILE__, __LINE__ );
        
	  if ( plane_id>0 ) {
	    // see if its in the daughter to mother map
	    auto it_showermap = _shower_daughter2mother.find( plane_id );
	    if ( it_showermap != _shower_daughter2mother.end() ) {
	      plane_id = (long)it_showermap->second;
	    }
	  }
        
	  if ( id_votes.find(plane_id)==id_votes.end() )
	    id_votes[plane_id] = 0;
	  id_votes[plane_id] += 1;
	}

	int maxid = 0;
	int nvotes = 0;
	for (auto it=id_votes.begin(); it!=id_votes.end(); it++) {
	  if ( it->first>0 && it->second>nvotes) {
	    nvotes = it->second;
	    maxid = it->first;
	  }
	}

	if ( maxid>0 )
	  nids++;
	
	matchdata._instance_id_v[itrip] = maxid;
      
      }//end of trips loop

      LARCV_NORMAL() << "=== (cryoid,tpcid)=(" << matchdata._cryoid << "," << matchdata._tpcid << ") =======" << std::endl;      
      LARCV_NORMAL() << "  number labeled: " << nids << " of " << matchdata._triplet_v.size() << std::endl;
    }//end of matchdata loop (over data from each tpc)
    
  }
  
  /**
   * @brief use ancestor ID image to label space points
   *
   * @param[in] ancestor_img_v Vector of Image2D which contain track id.
   *
   */
  void PrepMatchTriplets::make_ancestorid_vector( const std::vector<larcv::Image2D>& ancestor_img_v )
  {

    for ( auto& matchdata : _match_triplet_v ) {
      
      matchdata._ancestor_id_v.resize( matchdata._triplet_v.size(), 0);

      std::vector< const larcv::Image2D* > pancestor_v
	= ublarcvapp::recotools::DetUtils::getTPCImages( ancestor_img_v, matchdata._tpcid, matchdata._cryoid );
      
      int nids = 0;
      int ntrue_triplets =0;
    
      for ( size_t itrip=0; itrip<matchdata._triplet_v.size(); itrip++ ) {

	if ( matchdata._truth_v[itrip]==0 ) {
	  matchdata._ancestor_id_v[itrip] = 0;
	  continue;
	}
        
	ntrue_triplets++;
      
	// for each triplet, we look for truth flows that connect the planes
	auto const& triplet = matchdata._triplet_v[itrip];
	std::vector<int> imgcoord = {0,0,0,triplet[3]};
	for (int p=0; p<3; p++ ) {
	  imgcoord[p] = matchdata._sparseimg_vv[p][triplet[p]].col;
	}

	std::map< int, int > id_votes;

	for (int p=0; p<3; p++ ) {
	  int plane_id = pancestor_v[p]->pixel( imgcoord[3], imgcoord[p], __FILE__, __LINE__ );
	  if ( id_votes.find(plane_id)==id_votes.end() )
	    id_votes[plane_id] = 0;
	  id_votes[plane_id] += 1;
	}

	int maxid = 0;
	int nvotes = 0;
	for (auto it=id_votes.begin(); it!=id_votes.end(); it++) {
	  if ( it->first>0 && it->second>nvotes) {
	    nvotes = it->second;
	    maxid = it->first;
	  }
	}
	
	if ( maxid>0 )
	  nids++;
	
	matchdata._ancestor_id_v[itrip] = maxid;
	
      }//end of trips loop
      
      LARCV_NORMAL() << "=== (cryoid,tpcid)=(" << matchdata._cryoid << "," << matchdata._tpcid << ") =======" << std::endl;
      LARCV_NORMAL() << "  number labeled: " << nids
		     << " of " << matchdata._triplet_v.size() << " total "
		     << " and " << ntrue_triplets << "true" << std::endl;
    }//end of matchdata (from different tpcs)
    
  }
  
  /**
   * @brief use segment image to label particle class
   *
   * The id numbers contained in segment image correspond to enum values of larcv::ROIType_t.
   * ROIType_t is found in larcv/core/DataFormat/DataFormatTyps.h
   * 
   * @param[in] segment_img_v Vector of Image2D which containing larcv particle IDs
   *
   */
  void PrepMatchTriplets::make_segmentid_vector( const std::vector<larcv::Image2D>& segment_img_v,
                                                 const std::vector<larcv::Image2D>& adc_v )
  {

    for ( auto& matchdata : _match_triplet_v ) {
      
      matchdata._pdg_v.resize( matchdata._triplet_v.size(), 0);
      matchdata._origin_v.resize( matchdata._triplet_v.size(), 0 );

      std::vector< const larcv::Image2D* > psegment_v
	= ublarcvapp::recotools::DetUtils::getTPCImages( segment_img_v, matchdata._tpcid, matchdata._cryoid );
      std::vector< const larcv::Image2D* > padc_v
	= ublarcvapp::recotools::DetUtils::getTPCImages( adc_v, matchdata._tpcid, matchdata._cryoid );
      
      
      int nids = 0;
    
      for ( size_t itrip=0; itrip<matchdata._triplet_v.size(); itrip++ ) {

	if ( matchdata._truth_v[itrip]==0 ) {
	  matchdata._pdg_v[itrip] = 0;
	  matchdata._origin_v[itrip] = 0;
	  continue;
	}
      
	// for each triplet, we look for truth flows that connect the planes
	auto const& triplet = matchdata._triplet_v[itrip];
	std::vector<int> imgcoord = {0,0,0,triplet[3]};
	for (int p=0; p<3; p++ ) {
	  imgcoord[p] = matchdata._sparseimg_vv[p][triplet[p]].col;
	}

	std::map< int, float > id_votes;
	int nsegvotes = 0;

	for (int p=0; p<3; p++ ) {
	  int plane_id = psegment_v[p]->pixel( imgcoord[3], imgcoord[p], __FILE__, __LINE__ );
	  float pixval = padc_v[p]->pixel( imgcoord[3], imgcoord[p], __FILE__, __LINE__ );
	  if ( id_votes.find(plane_id)==id_votes.end() )
	    id_votes[plane_id] = 0;
	  //id_votes[plane_id] += pixval;
	  id_votes[plane_id] += 1;
	  if ( plane_id>=(int)larcv::kROIEminus )
	    nsegvotes++;
	}

	// also check the instance image for an id
	if ( matchdata._instance_id_v.size()>itrip ) {
	  int iid = matchdata._instance_id_v[itrip];
	  auto it_class = _instance2class_map.find( iid );
	  if ( it_class!=_instance2class_map.end() ) {
	    int pid = it_class->second;
	    if ( id_votes.find(pid)==id_votes.end() )
	      id_votes[pid] = 0;
	    id_votes[pid] += 10; // basically overrides segment image
	  }
	}
      
	int maxid = 0;
	float nvotes = 0;
	for (auto it=id_votes.begin(); it!=id_votes.end(); it++) {
	  if ( it->first>0 && it->second>0 && (it->second>nvotes) ) {
	    nvotes = it->second;
	    maxid = it->first;
	  }
	}
      
	if ( maxid>0 )
	  nids++;

	matchdata._pdg_v[itrip] = maxid;
	
	if ( nsegvotes>2 ) // need 3 planes with segment data (no dead wires on the segment images)
	  matchdata._origin_v[itrip] = 1;
	else
	  matchdata._origin_v[itrip] = 0;
      }//end of trips loop
    
      LARCV_NORMAL() << "=== (cryoid,tpcid)=(" << matchdata._cryoid << "," << matchdata._tpcid << ") =======" << std::endl;      
      LARCV_NORMAL() << "  number labeled: " << nids << " of " << matchdata._triplet_v.size() << std::endl;
      
    }//end of matchdata loop (over data from tpcs)
    
  }
  
  /**
   * @brief use mcreco data to label spacepoint origin
   *
   * Set flag as:
   *  origin=0 : unknown
   *  origin=1 : neutrino
   *  origin=2 : cosmic
   * 
   * @param[in] segment_img_v Vector of Image2D which containing larcv particle IDs
   *
   */
  void PrepMatchTriplets::make_origin_vector_frommcreco( larlite::storage_manager& ioll )    
  {

    // need instance to ancestor map
    std::map< int, int > trackid_to_ancestor;
    std::map< int, int > ancestor_to_origin;
    larlite::event_mctrack* ev_mctrack
      = (larlite::event_mctrack*)ioll.get_data( larlite::data::kMCTrack, "mcreco");
    for ( auto const& mctrack : *ev_mctrack ) {
      trackid_to_ancestor[(int)mctrack.TrackID()] = (int)mctrack.AncestorTrackID();
      if ( mctrack.TrackID()==mctrack.AncestorTrackID() ) {
	ancestor_to_origin[(int)mctrack.TrackID()] = mctrack.Origin();
      }
    }
    
    for ( auto& matchdata : _match_triplet_v ) {
    
      matchdata._origin_v.resize( matchdata._triplet_v.size(), 0 );
    
      if ( matchdata._ancestor_id_v.size() != matchdata._triplet_v.size() ) {
	throw std::runtime_error("Ancestor ID vector not filled yet. Run make_ancestorid_vector first.");
      }

    
      int nids = 0;
    
      for ( size_t itrip=0; itrip<matchdata._triplet_v.size(); itrip++ ) {
	
	if ( matchdata._truth_v[itrip]==0 ) {
	  matchdata._origin_v[itrip] = 0;
	  continue;
	}

	int trackid = matchdata._instance_id_v[itrip];
	auto it_t2a = trackid_to_ancestor.find( trackid );
	if ( it_t2a!=trackid_to_ancestor.end() ) {
	  // found ancestor label
	  int aid = it_t2a->second;
	  auto it_a2o = ancestor_to_origin.find(aid);
	  if ( it_a2o!=ancestor_to_origin.end() ) {
	    matchdata._origin_v[itrip] = it_a2o->second;
	    nids++;
	  }
	}
      
      }//end of trips loop
      
      LARCV_NORMAL() << "=== (cryoid,tpcid)=(" << matchdata._cryoid << "," << matchdata._tpcid << ") =======" << std::endl;    
      LARCV_NORMAL() << "  number labeled: " << nids << " of " << matchdata._triplet_v.size() << std::endl;
    }//matchdata loop
    
  }
  
}  
}
