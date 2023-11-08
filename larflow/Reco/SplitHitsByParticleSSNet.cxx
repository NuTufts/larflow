#include "SplitHitsByParticleSSNet.h"

#include "larcv/core/DataFormat/EventImage2D.h"
#include "larcv/core/DataFormat/EventSparseImage.h"

#include <ctime>

namespace larflow {
namespace reco {


  /**
   * @brief Process event data in the larcv and larlite IO managers
   *
   * @param[in] iolcv LArCV IO manager
   * @param[in] ioll  larlite IO manager
   */
  void SplitHitsByParticleSSNet::process( larcv::IOManager& iolcv,
                                          larlite::storage_manager& ioll )
  {

    larcv::EventImage2D* ev_wholeimage
      = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, "wire" );
    auto const& wholeview_v = ev_wholeimage->as_vector();
    
    larcv::EventSparseImage* ev_ssnet
      = (larcv::EventSparseImage*)iolcv.get_data( larcv::kProductSparseImage,
                                                  _input_ssnet_tree_name );
    auto const& sparseimg_v = ev_ssnet->SparseImageArray();
    LARCV_INFO() << "number of sparse images: " << sparseimg_v.size() << std::endl;
           
    larcv::EventImage2D* ev_fiveparticle
      = ( larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, "fivepidssn" );

    std::vector< larcv::Image2D > pid_v;
    
    for ( size_t p=0; p<3; p++ ) {
      
      const larcv::ImageMeta& meta = wholeview_v.at(p).meta();

      larcv::Image2D pid(meta);
      pid.paint(0);

      if (sparseimg_v.size()>0) {
        auto& spimg = sparseimg_v.at(p);
      
        int nfeatures = spimg.nfeatures();
        int stride = nfeatures+2;
        int npts = spimg.pixellist().size()/stride;
        auto const& spmeta = spimg.meta(0);
        
        for (int ipt=0; ipt<npts; ipt++) {
          int row = spimg.pixellist().at( ipt*stride+0 );
          int col = spimg.pixellist().at( ipt*stride+1 );
          
          int xrow = meta.row( spmeta.pos_y( row ) );
          int xcol = meta.col( spmeta.pos_x( col ) );
          
          int maxpid = -1;
          float maxscore = -1;
          for (int i=0; i<5; i++) {
            float score = spimg.pixellist().at( ipt*stride+2+i );
            if ( score>maxscore ) {
              maxscore = score;
              maxpid   = i;
            }
          }
        
          // float hip = spimg.pixellist().at( ipt*stride+2 );
          // float mip = spimg.pixellist().at( ipt*stride+3 );
          // float shr = spimg.pixellist().at( ipt*stride+4 );
          // float dlt = spimg.pixellist().at( ipt*stride+5 );
          // float mic = spimg.pixellist().at( ipt*stride+6 );
          
          pid.set_pixel( xrow, xcol, maxpid+1 );
        }//end of point loop
      }//end of if five particle ssn data exists
      
      pid_v.emplace_back( std::move(pid) );
    }//end of plane loop

    ev_fiveparticle->Emplace( std::move(pid_v) );
    
  }
 
  
}
}
