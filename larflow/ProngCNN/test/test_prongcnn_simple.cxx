// Simple test program for ProngCNN interface
// This program tests loading the model and basic functionality

#include <iostream>
#include <string>
#include <vector>

#include "TVector3.h"
#include "larflow/ProngCNN/ProngCNNInterface.h"
#include "larlite/DataFormat/larflowcluster.h"
#include "larcv/core/DataFormat/Image2D.h"

void print_usage() {
    std::cout << "Usage: test_prongcnn_simple [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -m, --model <path>  Path to ProngCNN model file (required)" << std::endl;
    std::cout << "  -d, --debug         Enable debug output" << std::endl;
    std::cout << "  -h, --help          Show this help message" << std::endl;
}

int main(int argc, char** argv) {
    
    // Parse command line arguments
    std::string model_file = "";
    bool debug = false;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-m" || arg == "--model") {
            if (i + 1 < argc) {
                model_file = argv[++i];
            }
        } else if (arg == "-d" || arg == "--debug") {
            debug = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage();
            return 0;
        }
    }
    
    // Validate required arguments
    if (model_file.empty()) {
        std::cerr << "Error: Model file path is required!" << std::endl;
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
    
    std::cout << "Model loaded successfully!" << std::endl;
    
    // Create dummy test data
    std::cout << "\nCreating test data..." << std::endl;
    
    // Create a test cluster with some dummy points
    larlite::larflowcluster test_cluster;
    for (int i = 0; i < 100; i++) {
        larlite::larflow3dhit hit;
        hit[0] = 100.0f + i*0.5f;  // x
        hit[1] = 50.0f + i*0.3f;   // y
        hit[2] = 200.0f + i*0.2f;  // z
        test_cluster.push_back(hit);
    }
    
    // Create test images (3 planes)
    std::vector<larcv::Image2D> adc_v;
    std::vector<larcv::Image2D> mask_v;
    
    for (int plane = 0; plane < 3; plane++) {
        // Create 512x512 images with some dummy data
        larcv::Image2D adc_img(512, 512);
        larcv::Image2D mask_img(512, 512);
        
        // Fill with some pattern
        for (int row = 0; row < 512; row++) {
            for (int col = 0; col < 512; col++) {
                float val = (row + col) % 100;
                adc_img.set_pixel(row, col, val);
                mask_img.set_pixel(row, col, val > 50 ? 1.0 : 0.0);
            }
        }
        
        adc_v.push_back(adc_img);
        mask_v.push_back(mask_img);
    }
    
    // Define crop point
    TVector3 crop_point(150.0, 100.0, 250.0);
    
    // Run inference
    std::cout << "\nRunning ProngCNN inference..." << std::endl;
    
    std::vector<float> pid_scores;
    std::vector<float> primary_parent_scores;
    int process;
    float purity, completeness;
    int nplanes_above;
    
    bool success = prongcnn.get_larpid_prong_scores(crop_point, test_cluster, adc_v, mask_v,
                                                     pid_scores, primary_parent_scores,
                                                     process, purity, completeness, nplanes_above);
    
    if (success) {
        std::cout << "\nInference successful!" << std::endl;
        std::cout << "Number of planes above threshold: " << nplanes_above << std::endl;
        
        if (pid_scores.size() >= 5) {
            std::cout << "\nPID Scores:" << std::endl;
            std::cout << "  Electron: " << pid_scores[0] << std::endl;
            std::cout << "  Photon:   " << pid_scores[1] << std::endl;
            std::cout << "  Muon:     " << pid_scores[2] << std::endl;
            std::cout << "  Pion:     " << pid_scores[3] << std::endl;
            std::cout << "  Proton:   " << pid_scores[4] << std::endl;
        }
        
        if (primary_parent_scores.size() >= 3) {
            std::cout << "\nProcess Scores:" << std::endl;
            std::cout << "  Primary:         " << primary_parent_scores[0] << std::endl;
            std::cout << "  Neutral Parent:  " << primary_parent_scores[1] << std::endl;
            std::cout << "  Charged Parent:  " << primary_parent_scores[2] << std::endl;
        }
        
        std::cout << "\nOther Metrics:" << std::endl;
        std::cout << "  Predicted Process: " << process << std::endl;
        std::cout << "  Purity:           " << purity << std::endl;
        std::cout << "  Completeness:     " << completeness << std::endl;
    } else {
        std::cout << "\nInference failed (possibly due to insufficient pixels above threshold)" << std::endl;
        std::cout << "Number of planes above threshold: " << nplanes_above << std::endl;
    }
    
    std::cout << "\nTest completed successfully!" << std::endl;
    
    return 0;
}