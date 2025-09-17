#include "TruthThrumuImageMaker.h"

#include "ublarcvapp/MCTools/MCPixelPGraph.h"
#include "larcv/core/DataFormat/EventImage2D.h"

namespace larflow {
namespace reco {

    void TruthThrumuImageMaker::process( larcv::IOManager& iolcv, larlite::storage_manager& ioll )
    {
        // get the wire plane signal image
        larcv::EventImage2D* ev_wire 
            = (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D, input_image_treename );
        auto const& wire_v = ev_wire->Image2DArray();

        // get the ancestor image
        larcv::EventImage2D* ev_ancestor 
            = (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D, "ancestor");
        auto const& ancestor_v = ev_ancestor->Image2DArray();

        // get output container
        larcv::EventImage2D* ev_thrumu
            = (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D, output_thrumu_treename);
        ev_thrumu->clear();

        // we use the MCPixelPGraph to get labels
        ublarcvapp::mctools::MCPixelPGraph mcpg;
        mcpg.buildgraphonly( ioll );

        // loop over the image planes
        size_t npixels_masked = 0;
        for ( size_t p=0; p<wire_v.size(); p++) {
            auto const& img_wire = wire_v.at(p);
            auto const& img_anc  = ancestor_v.at(p);

            larcv::Image2D thrumu( img_wire ); // start as a copy of the wire image

            auto const& pixdata_v = img_wire.as_vector();
            auto const& ancestor_labels_v = img_anc.as_vector(); 
            auto& thrumu_pixels_v = thrumu.as_mod_vector();

            for (size_t ipix=0; ipix<pixdata_v.size(); ipix++) {
                // ignore below threshold pixels
                if (pixdata_v.at(ipix)<10)
                    continue;

                int ancestor_label = ancestor_labels_v.at(ipix);
                // get origin using the MCPixelPGraph

                auto pnode = mcpg.findTrackID( ancestor_label );
                if ( pnode!=nullptr ) {
                    if ( pnode->origin==1 ) {
                        // if has the neutrino origin flag, we mask out the thrumu image pixel
                        thrumu_pixels_v.at(ipix) = 0;
                        npixels_masked++;
                    }
                }

            }//end of loop over image pixels

            ev_thrumu->Emplace( std::move(thrumu) );
        }//end of loop over planes

        LARCV_NORMAL() << "Masked " << npixels_masked << " pixels across the three planes." << std::endl;

    }


}
}