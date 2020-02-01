#include "WireOverlap.h"

#include <iostream>

#include "larlite/core/LArUtil/Geometry.h"

namespace larflow {

  bool WireOverlap::_isbuilt = false;
  std::vector< std::vector<int> >   WireOverlap::_wire_targetoverlap[6];
  std::vector< std::vector<int> >   WireOverlap::_wire_otheroverlap[6];
  std::map< std::pair<int,int>, int > WireOverlap::_planeflow2mapindex;

  WireOverlap::WireOverlap() {};

  /**
   * 
   * build what wires can overlap with each other
   *
   */
  void WireOverlap::_build() {

    if (_isbuilt) return;
   
    std::clock_t start = std::clock();
    
    const larutil::Geometry* geo = larutil::Geometry::GetME();

    float dt_intersect = 0.;
    
    // build wire overlap data
    int iflow = 0;
    for ( int srcplane=0; srcplane<3; srcplane++ ) {
      for (int tarplane=0; tarplane<3; tarplane++) {
        if ( tarplane==srcplane ) continue;
        std::cout << "[WireOverlap::build] constructing overlap maps for flow[" << iflow << "]: "
                  << "planes [" << srcplane << "] -> [" << tarplane << "] "
                  << std::endl;
        _wire_targetoverlap[iflow].resize( geo->Nwires(srcplane) );
        _wire_otheroverlap[iflow].resize(  geo->Nwires(srcplane) );
        
        int otherplane = -1;
        switch ( srcplane ) {
        case 0:
          otherplane = ( tarplane==1 ) ? 2 : 1;
          break;
        case 1:
          otherplane = ( tarplane==0 ) ? 2 : 0;
          break;
        case 2:
          otherplane = ( tarplane==0 ) ? 1 : 0;
          break;
        }
        
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
          _wire_targetoverlap[iflow][isrc].reserve( nwires );
          _wire_otheroverlap[iflow][isrc].reserve( nwires );
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
              if ( otherwire>=0 && otherwire<geo->Nwires(otherplane) ) {
                _wire_targetoverlap[iflow][isrc].push_back( tarwire );                    
                _wire_otheroverlap[iflow][isrc].push_back(  otherwire );
              }
            }

          }//loop over scan range

        }//loop over source wires

        _planeflow2mapindex[ std::pair<int,int>(srcplane,tarplane) ] = iflow;
        _planeflow2mapindex[ std::pair<int,int>(tarplane,srcplane) ] = iflow;        
        // increment flow        
        iflow++;
      }

      _isbuilt = true;
    }//end of loop over source planes
    std::clock_t end = std::clock();
    std::cout << "[wireOverlap] elapsed " << float(end-start)/float(CLOCKS_PER_SEC) << " secs "
              << " (intersect=" << dt_intersect << " secs)"
              << std::endl;
  }

  /** 
   * given a wire on the a defined source plane, 
   * what wires in the target plane, and other planes are intersected
   *
   * @param[in] sourceplane The index of the source plane. {0:U,1:V,2:Y}
   * @param[in] targetplane The index of the target plane. {0:U,1:V,2:Y}
   * @param[in] source_wire The index of the wire in the source plane.
   *
   * @param[out] double vector with outer index being the plane {target,other}, 
   *             the inner vector is a list of wire IDs from each plane
   *
   */
  std::vector< std::vector<int> > WireOverlap::getOverlappingWires( int sourceplane, int targetplane, int source_wire ) {
    
    if ( !_isbuilt ) _build();
    
    int iflow = _planeflow2mapindex[ std::pair<int,int>(sourceplane,targetplane) ];
    std::vector< std::vector<int> > overlap(2);
    overlap[0] = _wire_targetoverlap[iflow][source_wire];
    overlap[1] = _wire_otheroverlap[iflow][source_wire];
    return overlap;
    
  }

  

}
