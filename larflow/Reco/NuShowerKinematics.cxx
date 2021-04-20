#include "NuShowerKinematics.h"

namespace larflow {
namespace reco {

  void NuShowerKinematics::analyze( larflow::reco::NuVertexCandidate& nuvtx,
                                    larflow::reco::NuSelectionVariables& nusel,
                                    larcv::IOManager& iolcv )
  {

    _shower_mom_v.clear();

    larcv::EventImage2D* ev_img =
      (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D, "wire");

    auto const& adc_v = ev_img->as_vector();

    _shower_mom_v.resize(adc_v.size());
    
    for (int ishower=0; ishower<(int)nuvtx.shower_v.size(); ishower++) {
      auto const& cluster = nuvtx.shower_v[ishower];
      auto const& trunk   = nuvtx.shower_trunk_v[ishower];
      auto const& pca     = nuvtx.shower_pcaxis_v[ishower];

      std::vector<float> pixsum_v = GetADCSum( cluster,
                                               adc_v,
                                               10.0, 2);

      LARCV_DEBUG() << "shower[" << ishower << "] pixsum=(" << pixsum_v[0] << "," << pixsum_v[1] << "," << pixsum_v[2] << ")" << std::endl;
      
      TVector3 showerdir = get_showerdir( trunk, nuvtx.pos );
      LARCV_DEBUG() << "shower[" << ishower << "] "
		    << "dir=(" << showerdir[0] << "," << showerdir[1] << "," << showerdir[2] <<")"
		    << std::endl;
      
      std::vector< TLorentzVector > mom_v(adc_v.size());
      for (int p=0; p<(int)adc_v.size(); p++) {
        float MeV = (400.0/100.0e3)*pixsum_v[p];
        mom_v[p].SetPxPyPzE( MeV*showerdir[0], MeV*showerdir[1], MeV*showerdir[2], MeV );

        _shower_mom_v[p].push_back( mom_v[p] );
      }
      LARCV_INFO() << "shower[" << ishower << "] "
                   << "E(plane)=(" << mom_v[0].E()  << "," << mom_v[1].E() << "," << mom_v[2].E() << ") "
                   << "dir=(" << showerdir[0] << "," << showerdir[1] << "," << showerdir[2] << ")"
                   << std::endl;      
    }//end of shower loop

    
  }

  std::vector<float>
  NuShowerKinematics::GetADCSum(const larlite::larflowcluster& shower,
                                const std::vector<larcv::Image2D>& wire_img,
                                const int threshold,
                                const int dpix )
  {
    
    //initialize output
    std::vector<float> sum_v(wire_img.size(),0);
    
    // / first turn to cluster
    //save all three planes
    std::vector<larlite::larflow3dhit>  shower_c = shower;
    
    // turn into 2d points (u,v,y,t)
    std::vector< const larcv::ImageMeta* > wire_meta_v;
    for (auto const& img : wire_img )
      wire_meta_v.push_back( &img.meta() );

    // store image of pixels we've already counted
    std::vector<larcv::Image2D> mask_v;
    for ( auto const& img : wire_img ) {
      larcv::Image2D mask( img.meta() );
      mask.paint(0);
      mask_v.emplace_back( std::move(mask) );
    }
    
    // loop over planes
    for (int p =0;p<(int)wire_img.size();p++){
      
      const larcv::ImageMeta& meta =wire_img.at(p).meta();
      auto& mask = mask_v[p];
      
      float sum =0;
      // make found list so we don't use the same point multiple times vector<(row,col)>
      // std::vector<std::vector<int>> foundpts;
      // loop over Hits
      float noutpix =0.0;
      for ( size_t ihit=0; ihit<shower_c.size(); ihit++ ) {
        int wire = shower_c[ihit].targetwire[p];
        int tick =  shower_c[ihit].tick;
        
        // add a check for projection failures
        if (tick <= meta.min_y()
	    || tick >= meta.max_y()
	    || wire<= meta.min_x()
	    || wire >= meta.max_x() ){
          noutpix +=1.0;
          continue;
        }
        
        int row = meta.row(tick);
        int col = meta.col(wire);

        // neighborhood loop
        for (int dr=-abs(dpix); dr<=abs(dpix); dr++) {

          int r = row+dr;
          if ( r<0 || r>=(int)meta.rows() )
            continue;
          for (int dc=-abs(dpix); dc<abs(dpix); dc++) {
            int c = col+dc;
            if ( c<0 || c>=(int)meta.cols() )
              continue;
            
            // check mask
            float maskval = mask.pixel(r,c,__FILE__,__LINE__);
            
            if ( maskval>0 )
              continue; // already counted
            
            int adcval =0;
            adcval = wire_img[p].pixel(r,c,__FILE__,__LINE__);
        
            if (adcval >threshold) {
              sum = sum+adcval;
              mask.set_pixel(r,c,1.0);
            }
            
          }//end of dc loop
        }//end of dr loop
        
      }//end of loop over Hits
      
      if ((1.0-float(noutpix)/float(shower_c.size()))<=.98)
        sum_v[p] = 0.0;
      else
        sum_v[p] = sum;
      
    }//end of loop over planes
    
    return sum_v;
  }

  TVector3 NuShowerKinematics::get_showerdir( const larlite::track& shower_trunk,
                                              const std::vector<float>& vtxpos )
  {
    TVector3 vecpos(vtxpos[0],vtxpos[1],vtxpos[2]);

    float end1 = (shower_trunk.LocationAtPoint(0)-vecpos).Mag();
    float end2 = (shower_trunk.LocationAtPoint(1)-vecpos).Mag();

    TVector3 dir;
    if ( end1<end2 )
      dir = shower_trunk.LocationAtPoint(1)-shower_trunk.LocationAtPoint(0);
    else
      dir = shower_trunk.LocationAtPoint(0)-shower_trunk.LocationAtPoint(1);
    float mag = dir.Mag();
    if ( mag>0 ) {
      for (int i=0; i<3; i++)
        dir[i] /= mag;
    }

    return dir;
  }

}
}
