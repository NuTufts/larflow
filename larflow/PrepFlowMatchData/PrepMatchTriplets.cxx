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

    for ( auto& matchdata : _match_triplet_v )
      matchdata.clear();
    
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
    //larcv::EventChStatus* ev_badch
    //= (larcv::EventChStatus*)iolcv.get_data(larcv::kProductChStatus,chstatus_producer);


    std::vector< larcv::Image2D > badch_v;
    // just blanks for now
    for ( auto const& adc : ev_adc->as_vector() ) {
      larcv::Image2D badch_blank(adc.meta());
      badch_blank.paint(0.0);
      badch_v.emplace_back( badch_blank );
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
	
	process_tpc_v2( tpc_adc_v, tpc_badch_v, 10.0, itpc, icryo );
	
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
      std::vector< std::vector<int> > plane_cols(3);
      for (int iplane=0; iplane<nplanes; iplane++) {
     	int splaneid = start_plane+iplane;
     	auto const& pimg = adc_v.at(splaneid);
	//std::cout << "iplane[" << iplane << "] icryo=" << icryo << " itpc=" << itpc << " iplaneid=" << planeid << std::endl;
     	for (int icol=0; icol<(int)pimg->meta().cols(); icol++) {
     	  if ( pimg->pixel(irow,icol)>=adc_threshold )
     	    plane_cols[iplane].push_back(icol);
     	}
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
	      if ( adc_v[start_plane+ipl3]->pixel(irow,otherwire)>=adc_threshold ) {

		std::vector<int> wirecoord(4,0);
		wirecoord[ipl1] = idx1;
		wirecoord[ipl2] = idx2;
		wirecoord[ipl3] = otherwire;
		wirecoord[3]    = (int)tick;

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
  
}  
}
