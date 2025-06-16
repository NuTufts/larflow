#ifndef __NUVERTEXCANDIDATE_HDF5_READER_H__
#define __NUVERTEXCANDIDATE_HDF5_READER_H__

#include <string>
#include <vector>
#include <memory>
#include "hdf5.h"
#include "hdf5_hl.h"

// Forward declarations
namespace larlite {
  class track;
  class larflowcluster;
  class pcaxis;
}

namespace larflow {
namespace reco {
  class NuVertexCandidate;
}
}

namespace larflow {
namespace hdf5utils {

  /**
   * @brief Class to read NuVertexCandidate data from HDF5 files
   * 
   * This class reads HDF5 files created by convert_nuvertex_root_to_hdf5.py
   * and reconstructs NuVertexCandidate objects from the stored data.
   */
  class NuVertexCandidateHDF5Reader {
  public:
    NuVertexCandidateHDF5Reader();
    ~NuVertexCandidateHDF5Reader();
    
    /**
     * @brief Open an HDF5 file for reading
     * @param filename Path to the HDF5 file
     * @return True if successful, false otherwise
     */
    bool open(const std::string& filename);
    
    /**
     * @brief Close the currently open file
     */
    void close();
    
    /**
     * @brief Check if a file is currently open
     * @return True if file is open
     */
    bool isOpen() const { return _file_id >= 0; }
    
    /**
     * @brief Get the number of entries in the file
     * @return Number of entries
     */
    size_t getNumEntries() const { return _num_entries; }
    
    /**
     * @brief Get the total number of vertices across all entries
     * @return Total number of vertices
     */
    size_t getTotalNumVertices() const { return _total_vertices; }
    
    /**
     * @brief Get the number of vertices in a specific entry
     * @param entry Entry index
     * @return Number of vertices in the entry
     */
    size_t getNumVertices(size_t entry) const;
    
    /**
     * @brief Read all vertices from a specific entry
     * @param entry Entry index
     * @param vertices Output vector of NuVertexCandidate objects
     * @return True if successful
     */
    bool readEntry(size_t entry, std::vector<larflow::reco::NuVertexCandidate>& vertices);
    
    /**
     * @brief Read a specific vertex from an entry
     * @param entry Entry index
     * @param vertex_idx Vertex index within the entry
     * @param vertex Output NuVertexCandidate object
     * @return True if successful
     */
    bool readVertex(size_t entry, size_t vertex_idx, larflow::reco::NuVertexCandidate& vertex);
    
    /**
     * @brief Get metadata about the source file
     * @return Source ROOT filename
     */
    std::string getSourceFile() const { return _source_file; }
    
    /**
     * @brief Get the name of the original ROOT tree
     * @return Tree name
     */
    std::string getTreeName() const { return _tree_name; }
    
  private:
    // HDF5 handles
    hid_t _file_id;
    hid_t _vertices_group_id;
    
    // Metadata
    size_t _num_entries;
    size_t _total_vertices;
    std::string _source_file;
    std::string _tree_name;
    std::string _branch_name;
    
    // Helper methods
    bool readMetadata();
    bool readBasicVertexInfo(hid_t vtx_group, larflow::reco::NuVertexCandidate& vertex);
    bool readScores(hid_t vtx_group, larflow::reco::NuVertexCandidate& vertex);
    bool readClusters(hid_t vtx_group, larflow::reco::NuVertexCandidate& vertex);
    bool readClusterPCA(hid_t vtx_group, larflow::reco::NuVertexCandidate& vertex);
    bool readTracks(hid_t vtx_group, larflow::reco::NuVertexCandidate& vertex);
    bool readShowers(hid_t vtx_group, larflow::reco::NuVertexCandidate& vertex);
    
    // Utility methods for reading HDF5 data
    bool readFloatVector(hid_t group, const char* name, std::vector<float>& vec);
    bool readIntVector(hid_t group, const char* name, std::vector<int>& vec);
    bool readDoubleVector(hid_t group, const char* name, std::vector<double>& vec);
    bool readFloatAttribute(hid_t obj, const char* name, float& value);
    bool readIntAttribute(hid_t obj, const char* name, int& value) const;
    bool readStringAttribute(hid_t obj, const char* name, std::string& value);
    
    // Methods to reconstruct complex objects
    std::unique_ptr<larlite::track> reconstructTrack(hid_t track_group);
    std::unique_ptr<larlite::larflowcluster> reconstructLarflowCluster(hid_t cluster_group);
    std::unique_ptr<larlite::pcaxis> reconstructPCAxis(hid_t pca_group);
  };

} // namespace util
} // namespace larflow

#endif