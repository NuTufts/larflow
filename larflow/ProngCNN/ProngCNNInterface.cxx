#include "ProngCNNInterface.h"

#include "larpid/data/ModelOutput.h"
#include "larpid/data/CropPixData_t.h"
#include "larpid/interface/LArPIDInterface.h"


namespace larflow {
namespace prongcnn {

    bool ProngCNNInterface::load_model( std::string model_filepath, bool fdebug )
    {

        try {
            _model.Initialize( model_filepath, fdebug );
        }
        catch (std::exception& e) {
            //LARCV_ERROR() << "Could not load model: " << e.what() << std::endl;
            throw;
        }

        return true;
    }



    bool ProngCNNInterface::get_larpid_prong_scores( const TVector3& cropPt,
        const larlite::larflowcluster& hitcluster,
        larcv::IOManager& iolcv,
        bool preserve_shower_pixels,
        std::vector<float>& pid_scores,
        std::vector<float>& primary_and_parent_scores,
        int& process,
        float& purity_score,
        float& completeness_score,
        int& nplanes_above )
    {

        

        std::vector< std::vector<larpid::data::CropPixData_t> > prong_vv
            = larpid::interface::make_prongCNN_input_sparse_images( iolcv, hitcluster, 
                cropPt, preserve_shower_pixels );

/*
        size_t nplanes = adc_v.size(); 
        bool thresholdPassInOne = false;
        bool thresholdPassInAll = true;
        nplanes_above = 0;
        completeness_score = 0;
        purity_score = 0.;
        process = -1;

        for(size_t p = 0; p < nplanes; ++p){
          if(prong_vv[p].size() >= fPixelThreshold) 
            nplanes_above++; 
        }

        if ( nplanes_above<2 ) {
            return false;
        }

        bool network_run = false;
        larpid::data::ModelOutput output;
        try {
            output = _model.run_inference(prong_vv);
            network_run = true;
        }
        catch (std::exception& e) {
            network_run = false;
            //LARCV_WARNING() << "Error running larpid: " << e.what() << std::endl;
        }

        if ( !network_run ) {
            return false;
        }


        // LARCV_INFO() << "Successfuly ran LArPID! Model outputs:" << std::endl;
        // LARCV_INFO() << "    PID: " << output.pid << std::endl;
        // LARCV_INFO() << "    Production process: " << output.process << std::endl;
        // LARCV_INFO() << "    completeness: " << output.completeness << std::endl;
        // LARCV_INFO() << "    purity: " << output.purity << std::endl;
        // LARCV_INFO() << "    electron_score: " << output.electron_score << std::endl;
        // LARCV_INFO() << "    photon_score: " << output.photon_score << std::endl;
        // LARCV_INFO() << "    muon_score: " << output.muon_score << std::endl;
        // LARCV_INFO() << "    pion_score: " << output.pion_score << std::endl;
        // LARCV_INFO() << "    proton_score: " << output.proton_score << std::endl;
        // LARCV_INFO() << "    primary_score: " << output.primary_score << std::endl;
        // LARCV_INFO() << "    neutralParent_score: " << output.neutralParent_score << std::endl;
        // LARCV_INFO() << "    chargedParent_score: " << output.chargedParent_score << std::endl;

        // Copy classification scores (electron, photon, muon, pion, proton)
        pid_scores.resize(5,0.0);
        if (output.classScores.size() >= 5) {
            for (int i = 0; i < 5; i++) {
                pid_scores[i] = output.classScores[i];
            }
        }

        process = output.predictedProcess;
        purity_score = output.purity;
        completeness_score = output.completeness;

        // Copy process scores (primary, neutral parent, charged parent)
        primary_and_parent_scores.resize(3,0.0);
        if (output.processScores.size() >= 3) {
            for (int i = 0; i < 3; i++) {
                primary_and_parent_scores[i] = output.processScores[i];
            }
        }
*/
        return true;
    }
                        
  
  
}
}
