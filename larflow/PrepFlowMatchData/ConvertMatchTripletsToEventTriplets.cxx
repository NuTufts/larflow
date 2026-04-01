#include "ConvertMatchTripletsToEventTriplets.h"
#include "PrepMatchTriplets.h"

namespace larflow {
namespace prep {

  void ConvertMatchTripletsToEventTriplets::convert(
    ublarcvapp::mctools::MCPixelLabelMaker & mclabelmaker,
    larflow::prep::PrepMatchTriplets& tripletmaker)
  {

    // clear existing data
    mclabelmaker._pixels_v.clear();

    auto& pixels = mclabelmaker._pixels_v;
    auto const& meta0 = tripletmaker._imgmeta_v.at(0);
    int nplanes = (int)tripletmaker._sparseimg_vv.size();

    for (size_t itrip = 0; itrip < tripletmaker._triplet_v.size(); itrip++) {

      // only keep true spacepoints
      if ( tripletmaker._truth_v[itrip] != 1 )
        continue;

      // extract image coordinates
      int row = -1;
      std::array<int,5> imgcoord;
      std::array<float,3> pixval;
      for (int ip = 0; ip < nplanes; ip++) {
        auto const& pix = tripletmaker._sparseimg_vv.at(ip).at( tripletmaker._triplet_v[itrip][ip] );
        imgcoord[ip] = pix.col; // wire number for this plane
        pixval[ip] = pix.val;   // ADC value
        if (ip == 0)
          row = pix.row;
      }
      int tick = meta0.pos_y( row );
      imgcoord[3] = row;
      imgcoord[4] = tick;

      if ( tick < (int)meta0.min_y() || tick > (int)meta0.max_y() )
        continue;

      // check for duplicate (u,v,y,row) coordinates and merge if found
      std::array<int,4> coord4 = { imgcoord[0], imgcoord[1], imgcoord[2], imgcoord[3] };
      auto it = pixels._imgcoord_to_tripindex.find( coord4 );
      if ( it != pixels._imgcoord_to_tripindex.end() ) {
        // merge into existing entry
        auto& existing = pixels._triplets_v[ it->second ];
        if ( (int)tripletmaker._instance_id_v.size() > (int)itrip && tripletmaker._instance_id_v[itrip] != 0 )
          existing.trackids.insert( (long)tripletmaker._instance_id_v[itrip] );
        if ( (int)tripletmaker._ancestor_id_v.size() > (int)itrip && tripletmaker._ancestor_id_v[itrip] != 0 )
          existing.aids.insert( (long)tripletmaker._ancestor_id_v[itrip] );
        if ( (int)tripletmaker._pdg_v.size() > (int)itrip )
          existing.pids.insert( tripletmaker._pdg_v[itrip] );
        if ( (int)tripletmaker._origin_v.size() > (int)itrip )
          existing.origin.insert( tripletmaker._origin_v[itrip] );
        continue;
      }

      // create new MCPixelLabels entry
      ublarcvapp::mctools::MCPixelLabels label;
      label.index = (long)pixels._triplets_v.size();
      label.imgcoord = imgcoord;

      // true position left as zeros -- not available from old production files
      label.pos = {0.0, 0.0, 0.0};

      // reconstructed position from wire intersection
      label.pos_reco[0] = tripletmaker._pos_v[itrip][0];
      label.pos_reco[1] = tripletmaker._pos_v[itrip][1];
      label.pos_reco[2] = tripletmaker._pos_v[itrip][2];

      // energy deposition not available from old files
      label.edep = {0.0, 0.0, 0.0};

      label.pixval = pixval;

      // MC truth labels
      if ( (int)tripletmaker._instance_id_v.size() > (int)itrip && tripletmaker._instance_id_v[itrip] != 0 )
        label.trackids.insert( (long)tripletmaker._instance_id_v[itrip] );
      if ( (int)tripletmaker._ancestor_id_v.size() > (int)itrip && tripletmaker._ancestor_id_v[itrip] != 0 )
        label.aids.insert( (long)tripletmaker._ancestor_id_v[itrip] );
      if ( (int)tripletmaker._pdg_v.size() > (int)itrip )
        label.pids.insert( tripletmaker._pdg_v[itrip] );
      if ( (int)tripletmaker._origin_v.size() > (int)itrip )
        label.origin.insert( tripletmaker._origin_v[itrip] );

      pixels._imgcoord_to_tripindex[ coord4 ] = label.index;
      pixels._triplets_v.push_back( label );

    }

  }

}
}
