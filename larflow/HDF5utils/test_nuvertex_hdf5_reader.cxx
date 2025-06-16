/**
 * Example program demonstrating how to use NuVertexCandidateHDF5Reader
 * 
 * Usage: ./test_nuvertex_hdf5_reader <input.h5>
 */

#include "NuVertexCandidateHDF5Reader.h"
#include "larflow/Reco/NuVertexCandidate.h"
#include <iostream>
#include <iomanip>
#include <cstdlib>

void printVertexInfo(const larflow::reco::NuVertexCandidate& vtx, size_t entry, size_t vtx_idx) {
  std::cout << "\n=== Entry " << entry << ", Vertex " << vtx_idx << " ===" << std::endl;
  std::cout << "Keypoint: " << vtx.keypoint_producer << " [" << vtx.keypoint_index << "]" << std::endl;
  std::cout << "Position: (" << vtx.pos[0] << ", " << vtx.pos[1] << ", " << vtx.pos[2] << ")" << std::endl;
  std::cout << "Row: " << vtx.row << ", Tick: " << vtx.tick << std::endl;
  std::cout << "Scores - Overall: " << vtx.score 
            << ", Max: " << vtx.maxScore 
            << ", Avg: " << vtx.avgScore 
            << ", Net: " << vtx.netScore 
            << ", NetNu: " << vtx.netNuScore << std::endl;
  
  std::cout << "Clusters: " << vtx.cluster_v.size() << std::endl;
  for (size_t i = 0; i < vtx.cluster_v.size(); ++i) {
    const auto& cluster = vtx.cluster_v[i];
    std::cout << "  Cluster " << i << ": " << cluster.producer 
              << " [" << cluster.index << "], "
              << "gap=" << cluster.gap << ", impact=" << cluster.impact 
              << ", npts=" << cluster.npts << std::endl;
  }
  
  std::cout << "Tracks: " << vtx.track_v.size() << std::endl;
  for (size_t i = 0; i < vtx.track_v.size(); ++i) {
    std::cout << "  Track " << i << ": " 
              << "len=" << (i < vtx.track_len_v.size() ? vtx.track_len_v[i] : -1)
              << ", KE_mu=" << (i < vtx.track_kemu_v.size() ? vtx.track_kemu_v[i] : -1)
              << ", KE_p=" << (i < vtx.track_keproton_v.size() ? vtx.track_keproton_v[i] : -1)
              << ", mu_id=" << (i < vtx.track_muid_v.size() ? vtx.track_muid_v[i] : -1)
              << ", p_id=" << (i < vtx.track_protonid_v.size() ? vtx.track_protonid_v[i] : -1)
              << std::endl;
  }
  
  std::cout << "Showers: " << vtx.shower_v.size() << std::endl;
  for (size_t i = 0; i < vtx.shower_v.size(); ++i) {
    std::cout << "  Shower " << i << ": " 
              << "hits=" << vtx.shower_v[i].size()
              << ", secondary=" << (i < vtx.shower_isSecondary_v.size() ? vtx.shower_isSecondary_v[i] : -1)
              << std::endl;
    if (i < vtx.shower_plane_pixsum_vv.size()) {
      std::cout << "    Plane pixel sums: ";
      for (auto pix : vtx.shower_plane_pixsum_vv[i]) {
        std::cout << pix << " ";
      }
      std::cout << std::endl;
    }
  }
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <input.h5>" << std::endl;
    return 1;
  }
  
  std::string input_file = argv[1];
  
  // Create reader
  larflow::hdf5utils::NuVertexCandidateHDF5Reader reader;
  
  // Open file
  std::cout << "Opening HDF5 file: " << input_file << std::endl;
  if (!reader.open(input_file)) {
    std::cerr << "Error: Failed to open file" << std::endl;
    return 1;
  }
  
  // Print file info
  std::cout << "\nFile Information:" << std::endl;
  std::cout << "Source ROOT file: " << reader.getSourceFile() << std::endl;
  std::cout << "Original tree: " << reader.getTreeName() << std::endl;
  std::cout << "Number of entries: " << reader.getNumEntries() << std::endl;
  std::cout << "Total vertices: " << reader.getTotalNumVertices() << std::endl;
  
  // Read first few entries as examples
  size_t max_entries = std::min(size_t(3), reader.getNumEntries());
  std::cout << "\nReading first " << max_entries << " entries..." << std::endl;
  
  for (size_t entry = 0; entry < max_entries; ++entry) {
    std::cout << "\n--- Entry " << entry << " ---" << std::endl;
    std::cout << "Number of vertices: " << reader.getNumVertices(entry) << std::endl;
    
    // Read all vertices in this entry
    std::vector<larflow::reco::NuVertexCandidate> vertices;
    if (reader.readEntry(entry, vertices)) {
      for (size_t vtx_idx = 0; vtx_idx < vertices.size(); ++vtx_idx) {
        printVertexInfo(vertices[vtx_idx], entry, vtx_idx);
      }
    } else {
      std::cerr << "Failed to read entry " << entry << std::endl;
    }
  }
  
  // Example: Read a specific vertex
  if (reader.getNumEntries() > 0 && reader.getNumVertices(0) > 0) {
    std::cout << "\n\nExample: Reading specific vertex (entry 0, vertex 0):" << std::endl;
    larflow::reco::NuVertexCandidate single_vertex;
    if (reader.readVertex(0, 0, single_vertex)) {
      printVertexInfo(single_vertex, 0, 0);
    }
  }
  
  // Close file
  reader.close();
  
  std::cout << "\nDone!" << std::endl;
  return 0;
}