// ROOT system includes must come first
#include "RVersion.h"
#include "Rtypes.h"
#include "TFile.h"
#include "TTree.h"
#include "TVector3.h"

#include <iostream>
#include <string>
#include <vector>

// LArFlow includes
#include "larflow/ProngCNN/ProngCNNInterface.h"

// LArCV includes
#include "larcv/core/Base/larcv_logger.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/EventImage2D.h"

// Larlite includes
#include "larlite/DataFormat/storage_manager.h"
#include "larlite/DataFormat/larflowcluster.h"
#include "larlite/DataFormat/track.h"
#include "larlite/DataFormat/shower.h"

void print_usage() {
    std::cout << "Usage: run_prongcnn [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -m, --model <path>        Path to ProngCNN model file (required)" << std::endl;
    std::cout << "  -i, --input-reco <path>   Input larlite file with reco data (required)" << std::endl;
    std::cout << "  -s, --input-supera <path> Input supera file with images (required)" << std::endl;
    std::cout << "  -o, --output <path>       Output ROOT file (default: prongcnn_output.root)" << std::endl;
    std::cout << "  -n, --nevents <num>       Number of events to process (default: all)" << std::endl;
    std::cout << "  -d, --debug               Enable debug output" << std::endl;
    std::cout << "  -h, --help                Show this help message" << std::endl;
}

int main(int argc, char** argv) {
    
    // Parse command line arguments
    std::string model_file = "";
    std::string input_reco_file = "";
    std::string input_supera_file = "";
    std::string output_file = "prongcnn_output.root";
    int nevents = -1;
    bool debug = false;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-m" || arg == "--model") {
            if (i + 1 < argc) {
                model_file = argv[++i];
            }
        } else if (arg == "-i" || arg == "--input-reco") {
            if (i + 1 < argc) {
                input_reco_file = argv[++i];
            }
        } else if (arg == "-s" || arg == "--input-supera") {
            if (i + 1 < argc) {
                input_supera_file = argv[++i];
            }
        } else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) {
                output_file = argv[++i];
            }
        } else if (arg == "-n" || arg == "--nevents") {
            if (i + 1 < argc) {
                nevents = std::atoi(argv[++i]);
            }
        } else if (arg == "-d" || arg == "--debug") {
            debug = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage();
            return 0;
        }
    }
    
    // Validate required arguments
    if (model_file.empty() || input_reco_file.empty() || input_supera_file.empty()) {
        std::cerr << "Error: Missing required arguments!" << std::endl;
        print_usage();
        return 1;
    }
    
    // Initialize ProngCNN interface
    std::cout << "Loading ProngCNN model from: " << model_file << std::endl;
    larflow::prongcnn::ProngCNNInterface prongcnn;
    if (!prongcnn.load_model(model_file, debug)) {
        std::cerr << "Failed to load ProngCNN model!" << std::endl;
        return 1;
    }
    
    // Set up input file managers
    larlite::storage_manager io_reco(larlite::storage_manager::kREAD);
    io_reco.add_in_filename(input_reco_file);
    io_reco.open();
    
    larcv::IOManager io_supera(larcv::IOManager::kREAD);
    io_supera.add_in_file(input_supera_file);
    io_supera.initialize();
    
    // Set up output ROOT file and tree
    TFile* outfile = new TFile(output_file.c_str(), "RECREATE");
    TTree* outtree = new TTree("prongcnn", "ProngCNN scores for neutrino candidates");
    
    // Output tree variables
    int run, subrun, event;
    int nvertices;
    std::vector<float> vertex_x, vertex_y, vertex_z;
    std::vector<int> ntracks, nshowers;
    
    // Track variables (max 100 tracks per event)
    const int kMaxTracks = 100;
    int track_vertex_idx[kMaxTracks];
    int track_idx[kMaxTracks];
    float track_pid_scores[kMaxTracks][5]; // electron, photon, muon, pion, proton
    float track_primary_score[kMaxTracks];
    float track_neutral_parent_score[kMaxTracks];
    float track_charged_parent_score[kMaxTracks];
    int track_process[kMaxTracks];
    float track_purity[kMaxTracks];
    float track_completeness[kMaxTracks];
    int track_nplanes_above[kMaxTracks];
    int track_classified[kMaxTracks];
    int ntracks_total;
    
    // Shower variables (max 100 showers per event)
    const int kMaxShowers = 100;
    int shower_vertex_idx[kMaxShowers];
    int shower_idx[kMaxShowers];
    float shower_pid_scores[kMaxShowers][5];
    float shower_primary_score[kMaxShowers];
    float shower_neutral_parent_score[kMaxShowers];
    float shower_charged_parent_score[kMaxShowers];
    int shower_process[kMaxShowers];
    float shower_purity[kMaxShowers];
    float shower_completeness[kMaxShowers];
    int shower_nplanes_above[kMaxShowers];
    int shower_classified[kMaxShowers];
    int nshowers_total;
    
    // Set up tree branches
    outtree->Branch("run", &run, "run/I");
    outtree->Branch("subrun", &subrun, "subrun/I");
    outtree->Branch("event", &event, "event/I");
    outtree->Branch("nvertices", &nvertices, "nvertices/I");
    outtree->Branch("vertex_x", &vertex_x);
    outtree->Branch("vertex_y", &vertex_y);
    outtree->Branch("vertex_z", &vertex_z);
    outtree->Branch("ntracks", &ntracks);
    outtree->Branch("nshowers", &nshowers);
    
    outtree->Branch("ntracks_total", &ntracks_total, "ntracks_total/I");
    outtree->Branch("track_vertex_idx", track_vertex_idx, "track_vertex_idx[ntracks_total]/I");
    outtree->Branch("track_idx", track_idx, "track_idx[ntracks_total]/I");
    outtree->Branch("track_pid_scores", track_pid_scores, "track_pid_scores[ntracks_total][5]/F");
    outtree->Branch("track_primary_score", track_primary_score, "track_primary_score[ntracks_total]/F");
    outtree->Branch("track_neutral_parent_score", track_neutral_parent_score, "track_neutral_parent_score[ntracks_total]/F");
    outtree->Branch("track_charged_parent_score", track_charged_parent_score, "track_charged_parent_score[ntracks_total]/F");
    outtree->Branch("track_process", track_process, "track_process[ntracks_total]/I");
    outtree->Branch("track_purity", track_purity, "track_purity[ntracks_total]/F");
    outtree->Branch("track_completeness", track_completeness, "track_completeness[ntracks_total]/F");
    outtree->Branch("track_nplanes_above", track_nplanes_above, "track_nplanes_above[ntracks_total]/I");
    outtree->Branch("track_classified", track_classified, "track_classified[ntracks_total]/I");
    
    outtree->Branch("nshowers_total", &nshowers_total, "nshowers_total/I");
    outtree->Branch("shower_vertex_idx", shower_vertex_idx, "shower_vertex_idx[nshowers_total]/I");
    outtree->Branch("shower_idx", shower_idx, "shower_idx[nshowers_total]/I");
    outtree->Branch("shower_pid_scores", shower_pid_scores, "shower_pid_scores[nshowers_total][5]/F");
    outtree->Branch("shower_primary_score", shower_primary_score, "shower_primary_score[nshowers_total]/F");
    outtree->Branch("shower_neutral_parent_score", shower_neutral_parent_score, "shower_neutral_parent_score[nshowers_total]/F");
    outtree->Branch("shower_charged_parent_score", shower_charged_parent_score, "shower_charged_parent_score[nshowers_total]/F");
    outtree->Branch("shower_process", shower_process, "shower_process[nshowers_total]/I");
    outtree->Branch("shower_purity", shower_purity, "shower_purity[nshowers_total]/F");
    outtree->Branch("shower_completeness", shower_completeness, "shower_completeness[nshowers_total]/F");
    outtree->Branch("shower_nplanes_above", shower_nplanes_above, "shower_nplanes_above[nshowers_total]/I");
    outtree->Branch("shower_classified", shower_classified, "shower_classified[nshowers_total]/I");
    
    // Event loop
    int total_events = io_reco.get_entries();
    if (nevents > 0 && nevents < total_events) {
        total_events = nevents;
    }
    
    std::cout << "Processing " << total_events << " events..." << std::endl;
    
    for (int ientry = 0; ientry < total_events; ientry++) {
        
        if (ientry % 10 == 0) {
            std::cout << "Processing event " << ientry << " / " << total_events << std::endl;
        }
        
        // Read data
        io_reco.go_to(ientry);
        io_supera.read_entry(ientry);
        
        // Get run/subrun/event info
        run = io_reco.run_id();
        subrun = io_reco.subrun_id();
        event = io_reco.event_id();
        
        // Get images
        auto ev_adc = (larcv::EventImage2D*)(io_supera.get_data(larcv::kProductImage2D, "wire"));
        auto ev_thrumu = (larcv::EventImage2D*)(io_supera.get_data(larcv::kProductImage2D, "thrumu"));
        
        if (!ev_adc || !ev_thrumu) {
            std::cerr << "Missing image data for event " << ientry << std::endl;
            continue;
        }
        
        std::vector<larcv::Image2D> adc_v;
        std::vector<larcv::Image2D> mask_v;
        for (auto const& img : ev_adc->Image2DArray()) {
            adc_v.push_back(img);
        }
        for (auto const& img : ev_thrumu->Image2DArray()) {
            mask_v.push_back(img);
        }
        
        // Get neutrino vertex candidates
        auto ev_vertices = (larlite::event_larflowcluster*)io_reco.get_data(larlite::data::kLArFlowCluster, "wcnuvertex");
        if (!ev_vertices) {
            std::cerr << "No vertex data for event " << ientry << std::endl;
            continue;
        }
        
        // Clear vectors
        vertex_x.clear();
        vertex_y.clear();
        vertex_z.clear();
        ntracks.clear();
        nshowers.clear();
        
        nvertices = ev_vertices->size();
        ntracks_total = 0;
        nshowers_total = 0;
        
        // Loop over vertices
        for (int ivtx = 0; ivtx < nvertices; ivtx++) {
            auto const& vtx_cluster = ev_vertices->at(ivtx);
            
            // Get vertex position (use centroid of cluster)
            float vtx_x = 0, vtx_y = 0, vtx_z = 0;
            int npts = 0;
            for (size_t ipt = 0; ipt < vtx_cluster.size(); ipt++) {
                vtx_x += vtx_cluster[ipt][0];
                vtx_y += vtx_cluster[ipt][1];
                vtx_z += vtx_cluster[ipt][2];
                npts++;
            }
            if (npts > 0) {
                vtx_x /= npts;
                vtx_y /= npts;
                vtx_z /= npts;
            }
            
            vertex_x.push_back(vtx_x);
            vertex_y.push_back(vtx_y);
            vertex_z.push_back(vtx_z);
            
            // Get tracks associated with this vertex
            std::string track_producer = "nutrack_" + std::to_string(ivtx);
            auto ev_tracks = (larlite::event_track*)io_reco.get_data(larlite::data::kTrack, track_producer);
            auto ev_track_clusters = (larlite::event_larflowcluster*)io_reco.get_data(larlite::data::kLArFlowCluster, track_producer);
            
            int n_vertex_tracks = 0;
            if (ev_tracks && ev_track_clusters) {
                n_vertex_tracks = ev_tracks->size();
                
                for (int itrk = 0; itrk < n_vertex_tracks && ntracks_total < kMaxTracks; itrk++) {
                    auto const& track = ev_tracks->at(itrk);
                    auto const& track_cluster = ev_track_clusters->at(itrk);
                    
                    // Get track endpoint
                    TVector3 track_end(track.End().X(), track.End().Y(), track.End().Z());
                    
                    // Run ProngCNN
                    std::vector<float> pid_scores;
                    std::vector<float> primary_parent_scores;
                    int process;
                    float purity, completeness;
                    int nplanes_above;
                    
                    bool success = prongcnn.get_larpid_prong_scores(track_end, track_cluster, adc_v, mask_v,
                                                                     pid_scores, primary_parent_scores,
                                                                     process, purity, completeness, nplanes_above);
                    
                    track_vertex_idx[ntracks_total] = ivtx;
                    track_idx[ntracks_total] = itrk;
                    track_classified[ntracks_total] = success ? 1 : 0;
                    
                    if (success) {
                        for (int i = 0; i < 5; i++) {
                            track_pid_scores[ntracks_total][i] = pid_scores[i];
                        }
                        track_primary_score[ntracks_total] = primary_parent_scores[0];
                        track_neutral_parent_score[ntracks_total] = primary_parent_scores[1];
                        track_charged_parent_score[ntracks_total] = primary_parent_scores[2];
                        track_process[ntracks_total] = process;
                        track_purity[ntracks_total] = purity;
                        track_completeness[ntracks_total] = completeness;
                        track_nplanes_above[ntracks_total] = nplanes_above;
                    } else {
                        for (int i = 0; i < 5; i++) {
                            track_pid_scores[ntracks_total][i] = -1;
                        }
                        track_primary_score[ntracks_total] = -1;
                        track_neutral_parent_score[ntracks_total] = -1;
                        track_charged_parent_score[ntracks_total] = -1;
                        track_process[ntracks_total] = -1;
                        track_purity[ntracks_total] = -1;
                        track_completeness[ntracks_total] = -1;
                        track_nplanes_above[ntracks_total] = nplanes_above;
                    }
                    
                    ntracks_total++;
                }
            }
            ntracks.push_back(n_vertex_tracks);
            
            // Get showers associated with this vertex
            std::string shower_producer = "nushower_" + std::to_string(ivtx);
            auto ev_showers = (larlite::event_shower*)io_reco.get_data(larlite::data::kShower, shower_producer);
            auto ev_shower_clusters = (larlite::event_larflowcluster*)io_reco.get_data(larlite::data::kLArFlowCluster, shower_producer);
            
            int n_vertex_showers = 0;
            if (ev_showers && ev_shower_clusters) {
                n_vertex_showers = ev_showers->size();
                
                for (int ishw = 0; ishw < n_vertex_showers && nshowers_total < kMaxShowers; ishw++) {
                    auto const& shower = ev_showers->at(ishw);
                    auto const& shower_cluster = ev_shower_clusters->at(ishw);
                    
                    // Get shower start point
                    TVector3 shower_start(shower.ShowerStart().X(), shower.ShowerStart().Y(), shower.ShowerStart().Z());
                    
                    // Run ProngCNN
                    std::vector<float> pid_scores;
                    std::vector<float> primary_parent_scores;
                    int process;
                    float purity, completeness;
                    int nplanes_above;
                    
                    bool success = prongcnn.get_larpid_prong_scores(shower_start, shower_cluster, adc_v, mask_v,
                                                                     pid_scores, primary_parent_scores,
                                                                     process, purity, completeness, nplanes_above);
                    
                    shower_vertex_idx[nshowers_total] = ivtx;
                    shower_idx[nshowers_total] = ishw;
                    shower_classified[nshowers_total] = success ? 1 : 0;
                    
                    if (success) {
                        for (int i = 0; i < 5; i++) {
                            shower_pid_scores[nshowers_total][i] = pid_scores[i];
                        }
                        shower_primary_score[nshowers_total] = primary_parent_scores[0];
                        shower_neutral_parent_score[nshowers_total] = primary_parent_scores[1];
                        shower_charged_parent_score[nshowers_total] = primary_parent_scores[2];
                        shower_process[nshowers_total] = process;
                        shower_purity[nshowers_total] = purity;
                        shower_completeness[nshowers_total] = completeness;
                        shower_nplanes_above[nshowers_total] = nplanes_above;
                    } else {
                        for (int i = 0; i < 5; i++) {
                            shower_pid_scores[nshowers_total][i] = -1;
                        }
                        shower_primary_score[nshowers_total] = -1;
                        shower_neutral_parent_score[nshowers_total] = -1;
                        shower_charged_parent_score[nshowers_total] = -1;
                        shower_process[nshowers_total] = -1;
                        shower_purity[nshowers_total] = -1;
                        shower_completeness[nshowers_total] = -1;
                        shower_nplanes_above[nshowers_total] = nplanes_above;
                    }
                    
                    nshowers_total++;
                }
            }
            nshowers.push_back(n_vertex_showers);
        }
        
        // Fill output tree
        outtree->Fill();
    }
    
    // Write and close files
    outfile->cd();
    outtree->Write();
    outfile->Close();
    
    io_reco.close();
    io_supera.finalize();
    
    std::cout << "Finished processing. Output saved to: " << output_file << std::endl;
    
    return 0;
}