#include "WireOverlap.h"

#include <iostream>
#include <ctime>

#include "larlite/LArUtil/Geometry.h"
#include "larflow/LArFlowConstants/LArFlowConstants.h"

namespace larflow {
namespace prep {

  bool WireOverlap::_isbuilt = false;
  std::vector< std::vector<int> >   WireOverlap::_wire_targetoverlap[6];
  std::vector< std::vector<int> >   WireOverlap::_wire_otheroverlap[6];
  std::map< std::pair<int,int>, int > WireOverlap::_planeflow2mapindex;

  /*
   * constructor
   */
  WireOverlap::WireOverlap() {};

  /**
   * 
   * \brief determine which wires overlap with one another
   *
   * Is never called directly. Run once `getOverlappingWires` is called.
   *
   */
  void WireOverlap::_build() {

    if (_isbuilt) return;
   
    std::clock_t start = std::clock();
    
    const larutil::Geometry* geo = larutil::Geometry::GetME();

    float dt_intersect = 0.;
    
    // build wire overlap data
    for ( int flowdir=0; flowdir<(int)larflow::kNumFlows; flowdir++ ) {

      int srcplane, tarplane;
      larflow::LArFlowConstants::getFlowPlanes((larflow::FlowDir_t)flowdir,srcplane,tarplane);
      int otherplane = larflow::LArFlowConstants::getOtherPlane(srcplane,tarplane);

      std::cout << "[WireOverlap::build] constructing overlap maps for flow[" << flowdir << "]: "
                << "planes [" << srcplane << "] -> [" << tarplane << ", " << otherplane << "] "
                << std::endl;
        
                
      _wire_targetoverlap[flowdir].resize( geo->Nwires(srcplane) );        
      _wire_otheroverlap[flowdir].resize(  geo->Nwires(srcplane) );
        
      for (int isrc=0; isrc<(int)geo->Nwires(srcplane); isrc++) {

        Double_t xyzstart[3];
        Double_t xyzend[3];
        geo->WireEndPoints( (UChar_t)srcplane, (UInt_t)isrc, xyzstart, xyzend );

        float u1 = geo->WireCoordinate( xyzstart, (UInt_t)tarplane );
        float u2 = geo->WireCoordinate( xyzend,   (UInt_t)tarplane );

        float umin = (u1<u2) ? u1 : u2;
        float umax = (u1>u2) ? u1 : u2;

        umin -= 5.0;
        umax += 5.0;

        if ( umin<0 ) umin = 0;
        if ( umax<0 ) umax = 0;

        if ( (int)umin>=geo->Nwires(tarplane) ) umin = (float)geo->Nwires(tarplane)-1;
        if ( (int)umax>=geo->Nwires(tarplane) ) umax = (float)geo->Nwires(tarplane)-1;
        
        int nwires = umax-umin+1;
        _wire_targetoverlap[flowdir][isrc].reserve( nwires );
        _wire_otheroverlap[flowdir][isrc].reserve( nwires );

        for (int i=0; i<nwires; i++) {
          int tarwire = (int)umin+i;
          UInt_t src_ch = geo->PlaneWireToChannel( (UInt_t)srcplane, (UInt_t)isrc );
          UInt_t tar_ch = geo->PlaneWireToChannel( (UInt_t)tarplane, (UInt_t)tarwire );
          
          
          Double_t y,z;
          //std::clock_t xs_start = std::clock();
          bool crosses  = geo->ChannelsIntersect( src_ch, tar_ch, y, z );
          //dt_intersect += float(std::clock()-xs_start)/float(CLOCKS_PER_SEC);
          
          if ( crosses ) {
            Double_t pos[3] = { 0, y, z };
            int otherwire = geo->WireCoordinate( pos, otherplane );

            // for debug
            // if ( srcplane==1 && tarplane==2 )
            //   std::cout << "[" << srcplane << "," << tarplane << "," << otherplane << "] "
            //             << "wire=(" << xyzstart[1] << "," << xyzstart[2] << ")->(" << xyzend[1] << "," << xyzend[2] << ") "
            //             << "channels intersecting: (" << isrc << "," << tarwire << ") "
            //             << "@ (y,z)=(" << y << "," << z << ") -> otherwire="
            //             << otherwire << std::endl;
            
            if ( otherwire>=0 && otherwire<geo->Nwires(otherplane) ) {
              _wire_targetoverlap[flowdir][isrc].push_back( tarwire );                    
              _wire_otheroverlap[flowdir][isrc].push_back(  otherwire );
            }
            

          }

        }//loop over scan range
        
      }//loop over source wires

      _isbuilt = true;
    }//end of loop over source planes
    std::clock_t end = std::clock();
    std::cout << "[wireOverlap] elapsed " << float(end-start)/float(CLOCKS_PER_SEC) << " secs "
              << " (intersect=" << dt_intersect << " secs)"
              << std::endl;
    //std::cin.get();
  }

  /** 
   * @brief given a wire on source plane, return set of wires overlapping in the two other planes
   *
   * @param[in] sourceplane The index of the source plane. {0:U,1:V,2:Y}
   * @param[in] targetplane The index of the target plane. {0:U,1:V,2:Y}
   * @param[in] source_wire The index of the wire in the source plane.
   *
   * @return nested vector with outer index being the plane {target,other}, 
   *         the inner vector is a list of wire IDs from each plane
   *
   */
  std::vector< std::vector<int> > WireOverlap::getOverlappingWires( int sourceplane, int targetplane, int source_wire ) {
    
    if ( !_isbuilt ) _build();
    
    //int iflow = _planeflow2mapindex[ std::pair<int,int>(sourceplane,targetplane) ];
    int iflow = (int)larflow::LArFlowConstants::getFlowDirection(sourceplane,targetplane);
    std::vector< std::vector<int> > overlap(2);
    overlap[0] = _wire_targetoverlap[iflow][source_wire];
    overlap[1] = _wire_otheroverlap[iflow][source_wire];
    return overlap;
    
  }

  
}
}
