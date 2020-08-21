#include "PrepSSNetTriplet.h"

#include "larcv/core/DataFormat/EventImage2D.h"
#include "ublarcvapp/MCTools/crossingPointsAnaMethods.h"
#include "ublarcvapp/MCTools/NeutrinoVertex.h"

namespace larflow {
namespace prep {

  PrepSSNetTriplet::~PrepSSNetTriplet()
  {
    if (_label_tree) delete _label_tree;
  }
  
  
  /**
   * uses truth images from iomanager to make labels and weights for proposed triplets
   *
   */
  void PrepSSNetTriplet::make_ssnet_labels( larcv::IOManager& iolcv,
                                            larlite::storage_manager& ioll,
                                            const larflow::prep::PrepMatchTriplets& tripletmaker ) {

    // get truth images
    larcv::EventImage2D* ev_segment =
      (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D,"segment");

    // get neutrino vertex position
    std::vector<int> vtx_imgcoord =
      ublarcvapp::mctools::NeutrinoVertex::getImageCoords(ioll);


    _run = iolcv.event_id().run();
    _subrun = iolcv.event_id().subrun();
    _event  = iolcv.event_id().event();
    
    make_trackshower_labels( ev_segment->Image2DArray(),
                             tripletmaker,
                             vtx_imgcoord );
  }

  /** 
   * make track/shower labels
   *
   * labels
   * -------
   * [0]: background
   * [1]: track
   * [2]: shower
   *
   * weights
   * --------
   * 0: if not true proposal
   * 100: if within 10 pixels of neutrino vertex pixel
   * 10: if a boundary pixel
   * 1: everything else
   *
   * labels stored in _trackshower_label_v
   * weights stored in _trackshower_weight_v
   * follows triplet index vector, larflow::PrepMatchTriplets::_triplet_v
   *
   * @param[in] segment_v Vector of particle-ID labeled truth image
   * @param[in] tripletmaker Class that made triplets and stores values
   * @param[in] vtx_imgcoords Vector of image coordinates for true neutrino vertex (u,v,y,tick)
   *
   */
  void PrepSSNetTriplet::make_trackshower_labels( const std::vector<larcv::Image2D>& segment_v,
                                                  const larflow::prep::PrepMatchTriplets& tripletmaker,
                                                  const std::vector<int>& vtx_imgcoord )
  {

    
    size_t ntriplets = tripletmaker._triplet_v.size();
    _trackshower_label_v.clear();
    _trackshower_label_v.resize( ntriplets, 0 );
    _trackshower_weight_v.clear();
    _trackshower_weight_v.resize( ntriplets, 1.0 );
    _trackshower_num_v.clear();
    _trackshower_num_v.resize(3,0.0);

    int vtxrow = -1;
    if ( vtx_imgcoord[3]>tripletmaker._imgmeta_v.front().min_y()
         && vtx_imgcoord[3]<tripletmaker._imgmeta_v.front().max_y()  ) {
      vtxrow = tripletmaker._imgmeta_v.front().row( vtx_imgcoord[3] ); // convert tick to row
    }

    for (int i=0; i<3; i++) _trackshower_num_v[i] = 0;
      
    for ( size_t itrip=0; itrip<ntriplets; itrip++ ) {
      const std::vector<int>& triplet = tripletmaker._triplet_v[itrip];

      // if the triplet is bad, leave the label as 0 (background)
      if ( tripletmaker._truth_v[itrip]==0 ) {
        _trackshower_label_v[itrip] = 0;
        _trackshower_weight_v[itrip] = 1.;
        _trackshower_num_v[0] += 1.0;
        continue;
      }
      
      // we get the column for each plane
      std::vector<int> imgcoord(4,0); //(u,v,y,row)
      for (size_t p=0; p<3; p++)
        imgcoord[p] = tripletmaker._sparseimg_vv[p][triplet[p]].col;
      //use the y-plane for the row, should be the same      
      imgcoord[3] = tripletmaker._sparseimg_vv[2][triplet[2]].row;

      // got to planes and vote on label and boundary status
      int ntrack  = 0;
      int nshower = 0;
      int isboundary = 0;
      int isvertex = 0;
      for ( size_t p=0; p<3; p++ ) {
        auto const& seg = segment_v[p];
        int label = seg.pixel( imgcoord[3], imgcoord[p], __FILE__, __LINE__ );
        // determine if boundary
        int nneigh = 0;
        int nsame  = 0;
        for (size_t dr=-1; dr<=1; dr++) {
          int row = imgcoord[3]+dr;
          if ( row<0 || row>=(int)seg.meta().rows() ) continue;
          for ( size_t dc=-1; dc<=1; dc++ ) {
            int col = imgcoord[p]+dc;
            if ( col<0 || col>=seg.meta().cols() ) continue;
            int neighbor_label = seg.pixel(row,col,__FILE__,__LINE__);
            if ( neighbor_label!=0 ) {
              nneigh++;
              if ( neighbor_label==label )
                nsame++;
            }
          }//end of dcol loop
        }//end of drow loop
        if (nneigh!=nsame)
          isboundary += 1;

        // determine track/shower label
        if (label==larcv::kROIEminus || label==larcv::kROIGamma)
          nshower++;
        else
          ntrack++;

        // determine if near vertex
        if ( vtxrow>=0 && abs(imgcoord[3]-vtxrow)<10 && abs(imgcoord[p]-vtx_imgcoord[p])<10 )
          isvertex++;
      }//end of plane loop


      // set label
      if ( ntrack>nshower ) {
        _trackshower_label_v[itrip] = 1;
        _trackshower_num_v[1] += 1.0;        
      }
      else {
        _trackshower_label_v[itrip] = 2;
        _trackshower_num_v[2] += 1.0;        
      }

      // set weight
      if ( isvertex==3 )
        _trackshower_weight_v[itrip] = 100.;
      else if ( isboundary>0 )
        _trackshower_weight_v[itrip] = 10.;
      else
        _trackshower_weight_v[itrip] = 1.;

    }//end of loop over proposed triplets

    if ( _label_tree )
      _label_tree->Fill();
  }

  /** 
   * create the ana tree where we'll save labels
   *
   */
  void PrepSSNetTriplet::defineAnaTree()
  {
    _label_tree = new TTree("ssnetlabels","SSNet Triplet labels and weights");
    _label_tree->Branch( "run",    &_run,    "run/I" );
    _label_tree->Branch( "subrun", &_subrun, "subrun/I" );
    _label_tree->Branch( "event",  &_event,  "event/I" );    
    _label_tree->Branch( "trackshower_label_v",  &_trackshower_label_v );
    _label_tree->Branch( "trackshower_weight_v", &_trackshower_weight_v );
    _label_tree->Branch( "trackshower_num_v",    &_trackshower_num_v );
  }

  void PrepSSNetTriplet::writeAnaTree()
  {
    if ( _label_tree ) _label_tree->Write();
  }
  
}
}
    
