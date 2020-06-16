#include "KeypointFilterByWCTagger.h"

#include "LArUtil/LArProperties.h"
#include "LArUtil/Geometry.h"
#include "DataFormat/pcaxis.h"

namespace larflow {
namespace reco {

  KeypointFilterByWCTagger::KeypointFilterByWCTagger()
    : larcv::larcv_base("KeypointFilterByWCTagger")
  {
    set_defaults();
  }


  void KeypointFilterByWCTagger::process( larcv::IOManager& iolcv,
                                          larlite::storage_manager& ioll )
  {

    // get adc images
    larcv::EventImage2D* ev_adc_v =
      (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, _input_adc_tree_name );
    const std::vector<larcv::Image2D>& adc_v = ev_adc_v->Image2DArray();    
    LARCV_INFO() << "Input wire images [" << _input_adc_tree_name << "]: " << adc_v.size() << std::endl;
    
    // get cosmic tagged image
    larcv::EventImage2D* ev_tagger =
      (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, _input_taggerimg_tree_name );
    const std::vector<larcv::Image2D>& tagged_v = ev_tagger->Image2DArray();
    LARCV_INFO() << "Input tagged images [" << _input_taggerimg_tree_name << "]: " << tagged_v.size() << std::endl;

    // get keypoint network image
    LARCV_INFO() << "Input keypoint tree: " << _input_keypoint_tree_name << std::endl;    
    larlite::event_larflow3dhit* ev_keypoint =
      (larlite::event_larflow3dhit*)ioll.get_data( larlite::data::kLArFlow3DHit, _input_keypoint_tree_name );
    larlite::event_pcaxis* ev_keypoint_pcaxis =
      (larlite::event_pcaxis*)ioll.get_data( larlite::data::kPCAxis, _input_keypoint_tree_name );

    // get larmatch hits to filter
    LARCV_INFO() << "Input larmatch hit tree: " << _input_larmatch_tree_name << std::endl;
    larlite::event_larflow3dhit* ev_larmatch =
      (larlite::event_larflow3dhit*)ioll.get_data( larlite::data::kLArFlow3DHit, _input_larmatch_tree_name );
    
    // get ssnet shower images
    larcv::EventImage2D* ev_ssnet_v[3] = {nullptr};
    for ( size_t p=0; p<3; p++ ) {
      char prodname[20];
      sprintf( prodname, "%s%d", _ssnet_stem_name.c_str(), (int)p );
      ev_ssnet_v[p] = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, prodname );
    }

    // collect shower images
    std::vector<const larcv::Image2D*> ssnet_showerimg_v;
    for ( size_t p=0; p<3; p++ )
      ssnet_showerimg_v.push_back(&(ev_ssnet_v[p]->Image2DArray()[0]));


    std::vector<int> kept_hit_v( ev_larmatch->size(), 0 );
    filter_larmatchhits_using_tagged_image( adc_v, tagged_v, ssnet_showerimg_v, *ev_larmatch, kept_hit_v );

    std::vector<int> kept_keypoint_v( ev_keypoint->size(), 0 );
    filter_keypoint_using_tagged_image( adc_v, tagged_v, ssnet_showerimg_v, *ev_keypoint, kept_keypoint_v );
    

    larlite::event_larflow3dhit* ev_keypoint_output = 
      (larlite::event_larflow3dhit*)ioll.get_data( larlite::data::kLArFlow3DHit, _output_keypoint_tree_name );
    larlite::event_pcaxis* ev_kpaxis_output = 
      (larlite::event_pcaxis*)ioll.get_data( larlite::data::kPCAxis, _output_keypoint_tree_name );
    larlite::event_larflow3dhit* ev_filteredhits_output = 
      (larlite::event_larflow3dhit*)ioll.get_data( larlite::data::kLArFlow3DHit, _output_filteredhits_tree_name );

    for ( size_t ikp=0; ikp<ev_keypoint->size(); ikp++ ) {
      if ( kept_keypoint_v[ikp]==1 )  {
        ev_keypoint_output->push_back( ev_keypoint->at(ikp) );
        ev_kpaxis_output->push_back( ev_keypoint_pcaxis->at(ikp) );
      }
    }

    for ( size_t ihit=0; ihit<ev_larmatch->size(); ihit++ ) {
      if ( kept_hit_v[ihit]==1 ) 
        ev_filteredhits_output->push_back( ev_larmatch->at(ihit) );
    }

    LARCV_INFO() << "num of keypoint hits: " << ev_keypoint_output->size() << " of " << ev_keypoint->size() << std::endl;
    LARCV_INFO() << "num of filtered hits: " << ev_filteredhits_output->size() << " of " << ev_larmatch->size() << std::endl;
    
  }


  /**
   * 
   *
   */
  void KeypointFilterByWCTagger::filter_larmatchhits_using_tagged_image( const std::vector<larcv::Image2D>& adc_v,
                                                                         const std::vector<larcv::Image2D>& tagged_v,
                                                                         const std::vector< const larcv::Image2D* >& shower_ssnet_v,
                                                                         const std::vector<larlite::larflow3dhit>& larmatch_v,
                                                                         std::vector<int>& kept_v )
  {

    kept_v.clear();
    kept_v.resize( larmatch_v.size(), 0 );
    
    for ( size_t ihit=0; ihit<larmatch_v.size(); ihit++ ) {

      auto const& hit = larmatch_v[ihit];
      //LARCV_DEBUG() << "hit[" << ihit << "] tick=" << hit.tick+1 << std::endl;
      //int tick = hit[0]/larutil::LArProperties::GetME()->DriftVelocity()/0.5 + 3200;
      if ( hit.tick<=adc_v[0].meta().min_y() || hit.tick>=adc_v[0].meta().max_y() )
        continue;
      int row = adc_v[0].meta().row( hit.tick+1, __FILE__, __LINE__ );      
      int nplanes_tagged = 0;
      for ( size_t p=0; p<adc_v.size(); p++ ) {
        int col = adc_v[p].meta().col( hit.targetwire[p], __FILE__, __LINE__ );
        int tagged = tagged_v[p].pixel( row, col );
        //LARCV_DEBUG() << " tagged[col " << col << "]=" << tagged;        
        if ( tagged>5 ) {
          // check if shower
          float shower_score = shower_ssnet_v[p]->pixel(row,col);
          //LARCV_DEBUG() << " showerscore=" << shower_score;        
          if ( shower_score<0.5 ) {
            nplanes_tagged++;
          }

        }        
      }
      if ( nplanes_tagged>=2 ) {
        kept_v[ihit] = 0;
      }
      else {
        kept_v[ihit] = 1;
      }
      //LARCV_DEBUG() << "kept=" << kept_v[ihit] << std::endl;      
    }
    
  }


  /**
   * 
   *
   */
  void KeypointFilterByWCTagger::filter_keypoint_using_tagged_image( const std::vector<larcv::Image2D>& adc_v,
                                                                     const std::vector<larcv::Image2D>& tagged_v,
                                                                     const std::vector< const larcv::Image2D* >& shower_ssnet_v,
                                                                     const std::vector<larlite::larflow3dhit>& keypoint_v,
                                                                     std::vector<int>& kept_v )
  {

    kept_v.clear();
    kept_v.resize( keypoint_v.size(), 0 );
    
    for ( size_t ihit=0; ihit<keypoint_v.size(); ihit++ ) {

      auto const& hit = keypoint_v[ihit];

      int tick = hit[0]/larutil::LArProperties::GetME()->DriftVelocity()/0.5 + 3200;
      if ( tick<=adc_v[0].meta().min_y() || tick>=adc_v[0].meta().max_y() )
        continue;
      
      int row = adc_v[0].meta().row( tick, __FILE__, __LINE__ );
      
      std::vector<double> dpos = { hit[0], hit[1], hit[2] };

      int nplanes_tagged = 0;
      for ( size_t p=0; p<adc_v.size(); p++ ) {

        int wire = larutil::Geometry::GetME()->NearestWire( dpos, p );
        
        int col = adc_v[p].meta().col( wire );

        int nshower = 0;
        int ntagged = 0;

        for (int dr=-5; dr<=5; dr++) {
          int r = row+dr;
          if ( r<0 || r>=(int)adc_v[p].meta().rows() ) continue;
          for (int dc=-5; dc<=5; dc++) {
            int c = col+dc;
            if ( c<0 || c>=(int)adc_v[p].meta().cols() ) continue;

            int tagged = tagged_v[p].pixel( r, c );
            if ( tagged>5 ) {
              // check if shower
              float shower_score = shower_ssnet_v[p]->pixel(r,c);

              if ( shower_score<0.25 )
                nshower++;
            }

          }//col neighborhood    
        }//row neighborhood


        if (ntagged>0 && nshower<=2 )
          nplanes_tagged++;
      }//end of planes
        
      if ( nplanes_tagged>=2 ) {
        kept_v[ihit] = 0;
      }
      else {
        kept_v[ihit] = 1;
      }
      
    }//end of hit loop  
    
  }
  
  
}
}


