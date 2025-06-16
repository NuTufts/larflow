#include "NuVertexCandidateHDF5Reader.h"
#include "larflow/Reco/NuVertexCandidate.h"
#include "larlite/DataFormat/track.h"
#include "larlite/DataFormat/larflowcluster.h"
#include "larlite/DataFormat/pcaxis.h"
#include "TLorentzVector.h"
#include <iostream>
#include <sstream>
#include <algorithm>

namespace larflow {
namespace hdf5utils {

NuVertexCandidateHDF5Reader::NuVertexCandidateHDF5Reader() 
  : _file_id(-1), 
  _vertices_group_id(-1), 
  _num_entries(0), 
  _total_vertices(0),
  _tree_name("")
{
}

NuVertexCandidateHDF5Reader::~NuVertexCandidateHDF5Reader() {
  close();
}

bool NuVertexCandidateHDF5Reader::open(const std::string& filename) {
  // Close any open file
  close();
  
  // Open HDF5 file
  _file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (_file_id < 0) {
    std::cerr << "Error: Cannot open HDF5 file: " << filename << std::endl;
    return false;
  }
  
  // Read metadata
  if (!readMetadata()) {
    close();
    return false;
  }
  
  // Open vertices group
  _vertices_group_id = H5Gopen2(_file_id, "/vertices", H5P_DEFAULT);
  if (_vertices_group_id < 0) {
    std::cerr << "Error: Cannot open /vertices group" << std::endl;
    close();
    return false;
  }
  
  return true;
}

void NuVertexCandidateHDF5Reader::close() {
  if (_vertices_group_id >= 0) {
    H5Gclose(_vertices_group_id);
    _vertices_group_id = -1;
  }
  if (_file_id >= 0) {
    H5Fclose(_file_id);
    _file_id = -1;
  }
  _num_entries = 0;
  _total_vertices = 0;
}

bool NuVertexCandidateHDF5Reader::readMetadata() {
  if (_file_id < 0) return false;
  
  // Read file attributes
  readStringAttribute(_file_id, "source_file", _source_file);
  readStringAttribute(_file_id, "tree_name", _tree_name);
  readStringAttribute(_file_id, "branch_name", _branch_name);
  
  int n_entries = 0;
  if (readIntAttribute(_file_id, "n_entries", n_entries)) {
    _num_entries = n_entries;
  }
  
  int total_vertices = 0;
  if (readIntAttribute(_file_id, "total_vertices", total_vertices)) {
    _total_vertices = total_vertices;
  }
  
  return true;
}

size_t NuVertexCandidateHDF5Reader::getNumVertices(size_t entry) const {
  if (!isOpen() || entry >= _num_entries) return 0;
  
  // Open entry group
  std::stringstream entry_name;
  entry_name << "entry_" << entry;
  hid_t entry_group = H5Gopen2(_vertices_group_id, entry_name.str().c_str(), H5P_DEFAULT);
  if (entry_group < 0) return 0;
  
  // Read n_vertices attribute
  int n_vertices = 0;
  readIntAttribute(entry_group, "n_vertices", n_vertices);
  
  H5Gclose(entry_group);
  return n_vertices;
}

bool NuVertexCandidateHDF5Reader::readEntry(size_t entry, std::vector<larflow::reco::NuVertexCandidate>& vertices) {
  if (!isOpen() || entry >= _num_entries) return false;
  
  vertices.clear();
  size_t n_vertices = getNumVertices(entry);
  
  for (size_t vtx_idx = 0; vtx_idx < n_vertices; ++vtx_idx) {
    larflow::reco::NuVertexCandidate vertex;
    if (readVertex(entry, vtx_idx, vertex)) {
      vertices.push_back(vertex);
    }
  }
  
  return true;
}

bool NuVertexCandidateHDF5Reader::readVertex(size_t entry, size_t vertex_idx, 
                                              larflow::reco::NuVertexCandidate& vertex) {
  if (!isOpen() || entry >= _num_entries) return false;
  
  // Open vertex group
  std::stringstream vtx_name;
  vtx_name << "entry_" << entry << "_vtx_" << vertex_idx;
  hid_t vtx_group = H5Gopen2(_vertices_group_id, vtx_name.str().c_str(), H5P_DEFAULT);
  if (vtx_group < 0) {
    std::cerr << "Error: Cannot open vertex group: " << vtx_name.str() << std::endl;
    return false;
  }
  
  // Read all vertex data
  bool success = true;
  success &= readBasicVertexInfo(vtx_group, vertex);
  success &= readScores(vtx_group, vertex);
  success &= readClusters(vtx_group, vertex);
  success &= readClusterPCA(vtx_group, vertex);
  success &= readTracks(vtx_group, vertex);
  success &= readShowers(vtx_group, vertex);
  
  H5Gclose(vtx_group);
  return success;
}

bool NuVertexCandidateHDF5Reader::readBasicVertexInfo(hid_t vtx_group, 
                                                       larflow::reco::NuVertexCandidate& vertex) {
  // Read attributes
  readStringAttribute(vtx_group, "keypoint_producer", vertex.keypoint_producer);
  readIntAttribute(vtx_group, "keypoint_index", vertex.keypoint_index);
  readIntAttribute(vtx_group, "keypoint_type", vertex.keypoint_type);
  readIntAttribute(vtx_group, "row", vertex.row);
  readIntAttribute(vtx_group, "tick", vertex.tick);
  
  // Read datasets
  readFloatVector(vtx_group, "pos", vertex.pos);
  readIntVector(vtx_group, "col_v", vertex.col_v);
  
  return true;
}

bool NuVertexCandidateHDF5Reader::readScores(hid_t vtx_group, 
                                              larflow::reco::NuVertexCandidate& vertex) {
  hid_t scores_group = H5Gopen2(vtx_group, "scores", H5P_DEFAULT);
  if (scores_group < 0) return false;
  
  readFloatAttribute(scores_group, "score", vertex.score);
  readFloatAttribute(scores_group, "maxScore", vertex.maxScore);
  readFloatAttribute(scores_group, "avgScore", vertex.avgScore);
  readFloatAttribute(scores_group, "netScore", vertex.netScore);
  readFloatAttribute(scores_group, "netNuScore", vertex.netNuScore);
  
  H5Gclose(scores_group);
  return true;
}

bool NuVertexCandidateHDF5Reader::readClusters(hid_t vtx_group, 
                                                larflow::reco::NuVertexCandidate& vertex) {
  hid_t clusters_group = H5Gopen2(vtx_group, "clusters", H5P_DEFAULT);
  if (clusters_group < 0) return false;
  
  int n_clusters = 0;
  readIntAttribute(clusters_group, "n_clusters", n_clusters);
  
  vertex.cluster_v.clear();
  vertex.cluster_v.reserve(n_clusters);
  
  for (int i = 0; i < n_clusters; ++i) {
    std::stringstream cluster_name;
    cluster_name << "cluster_" << i;
    hid_t cluster_grp = H5Gopen2(clusters_group, cluster_name.str().c_str(), H5P_DEFAULT);
    if (cluster_grp < 0) continue;
    
    larflow::reco::NuVertexCandidate::VtxCluster_t cluster;
    readStringAttribute(cluster_grp, "producer", cluster.producer);
    readIntAttribute(cluster_grp, "index", cluster.index);
    readFloatAttribute(cluster_grp, "gap", cluster.gap);
    readFloatAttribute(cluster_grp, "impact", cluster.impact);
    readIntAttribute(cluster_grp, "npts", cluster.npts);
    
    int type_int = 0;
    readIntAttribute(cluster_grp, "type", type_int);
    cluster.type = static_cast<larflow::reco::NuVertexCandidate::ClusterType_t>(type_int);
    
    readFloatVector(cluster_grp, "dir", cluster.dir);
    readFloatVector(cluster_grp, "pos", cluster.pos);
    
    vertex.cluster_v.push_back(cluster);
    H5Gclose(cluster_grp);
  }
  
  H5Gclose(clusters_group);
  return true;
}

bool NuVertexCandidateHDF5Reader::readClusterPCA(hid_t vtx_group, 
                                                  larflow::reco::NuVertexCandidate& vertex) {
  hid_t pca_group = H5Gopen2(vtx_group, "cluster_pca", H5P_DEFAULT);
  if (pca_group < 0) return false;
  
  int n_pca = 0;
  readIntAttribute(pca_group, "n_pca", n_pca);
  
  vertex.cluster_pca_v.clear();
  vertex.cluster_pca_v.reserve(n_pca);
  
  for (int i = 0; i < n_pca; ++i) {
    std::stringstream pca_name;
    pca_name << "pca_" << i;
    hid_t pca_grp = H5Gopen2(pca_group, pca_name.str().c_str(), H5P_DEFAULT);
    if (pca_grp < 0) continue;
    
    auto pca = reconstructPCAxis(pca_grp);
    if (pca) {
      vertex.cluster_pca_v.push_back(*pca);
    }
    
    H5Gclose(pca_grp);
  }
  
  H5Gclose(pca_group);
  return true;
}

bool NuVertexCandidateHDF5Reader::readTracks(hid_t vtx_group, 
                                              larflow::reco::NuVertexCandidate& vertex) {
  hid_t tracks_group = H5Gopen2(vtx_group, "tracks", H5P_DEFAULT);
  if (tracks_group < 0) return false;
  
  int n_tracks = 0;
  readIntAttribute(tracks_group, "n_tracks", n_tracks);
  
  // Clear track vectors
  vertex.track_v.clear();
  vertex.track_hitcluster_v.clear();
  vertex.track_len_v.clear();
  vertex.track_dir_v.clear();
  vertex.track_kemu_v.clear();
  vertex.track_keproton_v.clear();
  vertex.track_pmu_v.clear();
  vertex.track_pproton_v.clear();
  vertex.track_muid_v.clear();
  vertex.track_protonid_v.clear();
  vertex.track_mu_vs_proton_llratio_v.clear();
  vertex.track_isSecondary_v.clear();
  
  if (n_tracks > 0) {
    // Read track property arrays
    readFloatVector(tracks_group, "track_len_v", vertex.track_len_v);
    // readFloatVector(tracks_group, "track_kemu_v", vertex.track_kemu_v);
    // readFloatVector(tracks_group, "track_keproton_v", vertex.track_keproton_v);
    // readFloatVector(tracks_group, "track_muid_v", vertex.track_muid_v);
    // readFloatVector(tracks_group, "track_protonid_v", vertex.track_protonid_v);
    // readFloatVector(tracks_group, "track_mu_vs_proton_llratio_v", vertex.track_mu_vs_proton_llratio_v);
    readIntVector(tracks_group, "track_isSecondary_v", vertex.track_isSecondary_v);
    
    // Read track directions (2D array)
    hid_t dir_dataset = H5Dopen2(tracks_group, "track_dir_v", H5P_DEFAULT);
    if (dir_dataset >= 0) {
      hid_t dataspace = H5Dget_space(dir_dataset);
      hsize_t dims[2];
      H5Sget_simple_extent_dims(dataspace, dims, NULL);
      
      std::vector<float> flat_dirs(dims[0] * dims[1]);
      H5Dread(dir_dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, flat_dirs.data());
      
      vertex.track_dir_v.resize(dims[0]);
      for (size_t i = 0; i < dims[0]; ++i) {
        vertex.track_dir_v[i].resize(dims[1]);
        for (size_t j = 0; j < dims[1]; ++j) {
          vertex.track_dir_v[i][j] = flat_dirs[i * dims[1] + j];
        }
      }
      
      H5Sclose(dataspace);
      H5Dclose(dir_dataset);
    }
    
    // Read track momenta (2D arrays)
    auto read4Vectors = [](hid_t group, const char* name, std::vector<TLorentzVector>& vec) {
      hid_t dataset = H5Dopen2(group, name, H5P_DEFAULT);
      if (dataset >= 0) {
        hid_t dataspace = H5Dget_space(dataset);
        hsize_t dims[2];
        H5Sget_simple_extent_dims(dataspace, dims, NULL);
        
        std::vector<float> flat_data(dims[0] * dims[1]);
        H5Dread(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, flat_data.data());
        
        vec.resize(dims[0]);
        for (size_t i = 0; i < dims[0]; ++i) {
          vec[i].SetPxPyPzE(flat_data[i*4+1], flat_data[i*4+2], 
                             flat_data[i*4+3], flat_data[i*4+0]);
        }
        
        H5Sclose(dataspace);
        H5Dclose(dataset);
      }
    };
    
    // read4Vectors(tracks_group, "track_pmu_v", vertex.track_pmu_v);
    // read4Vectors(tracks_group, "track_pproton_v", vertex.track_pproton_v);
    
    // Read individual tracks
    vertex.track_v.reserve(n_tracks);
    for (int i = 0; i < n_tracks; ++i) {
      std::stringstream track_name;
      track_name << "track_" << i;
      hid_t track_grp = H5Gopen2(tracks_group, track_name.str().c_str(), H5P_DEFAULT);
      if (track_grp < 0) continue;
      
      auto track = reconstructTrack(track_grp);
      if (track) {
        vertex.track_v.push_back(*track);
      }
      
      H5Gclose(track_grp);
    }
    
    // Initialize empty hitcluster vector to match track count
    vertex.track_hitcluster_v.resize(n_tracks);
  }
  
  H5Gclose(tracks_group);
  return true;
}

bool NuVertexCandidateHDF5Reader::readShowers(hid_t vtx_group, 
                                               larflow::reco::NuVertexCandidate& vertex) {
  hid_t showers_group = H5Gopen2(vtx_group, "showers", H5P_DEFAULT);
  if (showers_group < 0) return false;
  
  int n_showers = 0;
  readIntAttribute(showers_group, "n_showers", n_showers);
  
  // Clear shower vectors
  vertex.shower_v.clear();
  vertex.shower_trunk_v.clear();
  vertex.shower_pcaxis_v.clear();
  vertex.shower_plane_pixsum_vv.clear();
  vertex.shower_plane_mom_vv.clear();
  vertex.shower_plane_dqdx_vv.clear();
  vertex.shower_isSecondary_v.clear();
  
  if (n_showers > 0) {
    readIntVector(showers_group, "shower_isSecondary_v", vertex.shower_isSecondary_v);
    
    // Read 2D arrays for shower plane properties
    auto read2DFloatArray = [this](hid_t group, const char* name, 
                                   std::vector<std::vector<float>>& vec) {
      hid_t dataset = H5Dopen2(group, name, H5P_DEFAULT);
      if (dataset >= 0) {
        hid_t dataspace = H5Dget_space(dataset);
        hsize_t dims[2];
        H5Sget_simple_extent_dims(dataspace, dims, NULL);
        
        std::vector<float> flat_data(dims[0] * dims[1]);
        H5Dread(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, flat_data.data());
        
        vec.resize(dims[0]);
        for (size_t i = 0; i < dims[0]; ++i) {
          vec[i].resize(dims[1]);
          for (size_t j = 0; j < dims[1]; ++j) {
            vec[i][j] = flat_data[i * dims[1] + j];
          }
        }
        
        H5Sclose(dataspace);
        H5Dclose(dataset);
      }
    };
    
    read2DFloatArray(showers_group, "shower_plane_pixsum_vv", vertex.shower_plane_pixsum_vv);
    read2DFloatArray(showers_group, "shower_plane_dqdx_vv", vertex.shower_plane_dqdx_vv);
    
    // Read 3D array for shower momenta
    hid_t mom_dataset = H5Dopen2(showers_group, "shower_plane_mom_vv", H5P_DEFAULT);
    if (mom_dataset >= 0) {
      hid_t dataspace = H5Dget_space(mom_dataset);
      int ndims = H5Sget_simple_extent_ndims(dataspace);
      hsize_t dims[3];
      H5Sget_simple_extent_dims(dataspace, dims, NULL);
      
      std::vector<float> flat_data(dims[0] * dims[1] * dims[2]);
      H5Dread(mom_dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, flat_data.data());
      
      vertex.shower_plane_mom_vv.resize(dims[0]);
      for (size_t i = 0; i < dims[0]; ++i) {
        vertex.shower_plane_mom_vv[i].resize(dims[1]);
        for (size_t j = 0; j < dims[1]; ++j) {
          size_t idx = (i * dims[1] + j) * 4;
          vertex.shower_plane_mom_vv[i][j].SetPxPyPzE(
            flat_data[idx+1], flat_data[idx+2], flat_data[idx+3], flat_data[idx+0]);
        }
      }
      
      H5Sclose(dataspace);
      H5Dclose(mom_dataset);
    }
    
    // Read individual showers
    vertex.shower_v.reserve(n_showers);
    vertex.shower_trunk_v.reserve(n_showers);
    vertex.shower_pcaxis_v.reserve(n_showers);
    
    for (int i = 0; i < n_showers; ++i) {
      std::stringstream shower_name;
      shower_name << "shower_" << i;
      hid_t shower_grp = H5Gopen2(showers_group, shower_name.str().c_str(), H5P_DEFAULT);
      if (shower_grp < 0) continue;
      
      // Read shower cluster
      auto shower = reconstructLarflowCluster(shower_grp);
      if (shower) {
        vertex.shower_v.push_back(*shower);
      }
      
      // Read shower trunk
      hid_t trunk_grp = H5Gopen2(shower_grp, "trunk", H5P_DEFAULT);
      if (trunk_grp >= 0) {
        auto trunk = reconstructTrack(trunk_grp);
        if (trunk) {
          vertex.shower_trunk_v.push_back(*trunk);
        }
        H5Gclose(trunk_grp);
      }
      
      // Read shower PCA
      hid_t pca_grp = H5Gopen2(shower_grp, "pcaxis", H5P_DEFAULT);
      if (pca_grp >= 0) {
        auto pca = reconstructPCAxis(pca_grp);
        if (pca) {
          vertex.shower_pcaxis_v.push_back(*pca);
        }
        H5Gclose(pca_grp);
      }
      
      H5Gclose(shower_grp);
    }
  }
  
  H5Gclose(showers_group);
  return true;
}

// Utility methods
bool NuVertexCandidateHDF5Reader::readFloatVector(hid_t group, const char* name, 
                                                   std::vector<float>& vec) {
  hid_t dataset = H5Dopen2(group, name, H5P_DEFAULT);
  if (dataset < 0) return false;
  
  hid_t dataspace = H5Dget_space(dataset);
  hsize_t dims[1];
  H5Sget_simple_extent_dims(dataspace, dims, NULL);
  
  vec.resize(dims[0]);
  H5Dread(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, vec.data());
  
  H5Sclose(dataspace);
  H5Dclose(dataset);
  return true;
}

bool NuVertexCandidateHDF5Reader::readIntVector(hid_t group, const char* name, 
                                                 std::vector<int>& vec) {
  hid_t dataset = H5Dopen2(group, name, H5P_DEFAULT);
  if (dataset < 0) return false;
  
  hid_t dataspace = H5Dget_space(dataset);
  hsize_t dims[1];
  H5Sget_simple_extent_dims(dataspace, dims, NULL);
  
  vec.resize(dims[0]);
  H5Dread(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, vec.data());
  
  H5Sclose(dataspace);
  H5Dclose(dataset);
  return true;
}

bool NuVertexCandidateHDF5Reader::readDoubleVector(hid_t group, const char* name, 
                                                    std::vector<double>& vec) {
  hid_t dataset = H5Dopen2(group, name, H5P_DEFAULT);
  if (dataset < 0) return false;
  
  hid_t dataspace = H5Dget_space(dataset);
  hsize_t dims[1];
  H5Sget_simple_extent_dims(dataspace, dims, NULL);
  
  vec.resize(dims[0]);
  H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, vec.data());
  
  H5Sclose(dataspace);
  H5Dclose(dataset);
  return true;
}

bool NuVertexCandidateHDF5Reader::readFloatAttribute(hid_t obj, const char* name, float& value) {
  hid_t attr = H5Aopen(obj, name, H5P_DEFAULT);
  if (attr < 0) return false;
  
  H5Aread(attr, H5T_NATIVE_FLOAT, &value);
  H5Aclose(attr);
  return true;
}

bool NuVertexCandidateHDF5Reader::readIntAttribute(hid_t obj, const char* name, int& value) const {
  hid_t attr = H5Aopen(obj, name, H5P_DEFAULT);
  if (attr < 0) return false;
  
  H5Aread(attr, H5T_NATIVE_INT, &value);
  H5Aclose(attr);
  return true;
}

bool NuVertexCandidateHDF5Reader::readStringAttribute(hid_t obj, const char* name, std::string& value) {
  hid_t attr = H5Aopen(obj, name, H5P_DEFAULT);
  if (attr < 0) return false;
  
  hid_t atype = H5Aget_type(attr);
  hid_t aspace = H5Aget_space(attr);
  
  // Check if it's a variable-length string
  if (H5Tis_variable_str(atype)) {
    char* rdata;
    H5Aread(attr, atype, &rdata);
    if (rdata) {
      value = std::string(rdata);
      H5free_memory(rdata);
    } else {
      value = "";
    }
  } else {
    // Fixed-length string
    size_t size = H5Tget_size(atype);
    char* buffer = new char[size + 1];
    H5Aread(attr, atype, buffer);
    buffer[size] = '\0';
    
    // Remove any null padding
    value = std::string(buffer);
    size_t nullpos = value.find('\0');
    if (nullpos != std::string::npos) {
      value = value.substr(0, nullpos);
    }
    delete[] buffer;
  }
  
  H5Sclose(aspace);
  H5Tclose(atype);
  H5Aclose(attr);
  return true;
}

std::unique_ptr<larlite::track> NuVertexCandidateHDF5Reader::reconstructTrack(hid_t track_group) {
  auto track = std::make_unique<larlite::track>();
  
  int id = 0, n_points = 0;
  float length = 0;
  readIntAttribute(track_group, "id", id);
  readIntAttribute(track_group, "n_points", n_points);
  readFloatAttribute(track_group, "length", length);
  
  track->set_track_id(id);
  
  // Read trajectory points if available
  hid_t points_dataset = H5Dopen2(track_group, "points", H5P_DEFAULT);
  if (points_dataset >= 0 && n_points > 0) {
    hid_t dataspace = H5Dget_space(points_dataset);
    hsize_t dims[2];
    H5Sget_simple_extent_dims(dataspace, dims, NULL);
    
    std::vector<float> flat_points(dims[0] * dims[1]);
    H5Dread(points_dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, flat_points.data());
    
    // Convert to track format
    for (size_t i = 0; i < dims[0]; ++i) {
      TVector3 pos(flat_points[i*3], flat_points[i*3+1], flat_points[i*3+2]);
      TVector3 mom(0, 0, 0); // Momentum not stored in HDF5
      track->add_vertex(pos);
      track->add_direction(mom);
    }
    
    H5Sclose(dataspace);
    H5Dclose(points_dataset);
  }
  
  return track;
}

std::unique_ptr<larlite::larflowcluster> NuVertexCandidateHDF5Reader::reconstructLarflowCluster(hid_t cluster_group) {
  auto cluster = std::make_unique<larlite::larflowcluster>();
  
  int n_hits = 0;
  readIntAttribute(cluster_group, "n_hits", n_hits);
  
  // Read hit positions if available
  hid_t hits_dataset = H5Dopen2(cluster_group, "hits", H5P_DEFAULT);
  if (hits_dataset >= 0 && n_hits > 0) {
    hid_t dataspace = H5Dget_space(hits_dataset);
    hsize_t dims[2];
    H5Sget_simple_extent_dims(dataspace, dims, NULL);
    
    std::vector<float> flat_hits(dims[0] * dims[1]);
    H5Dread(hits_dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, flat_hits.data());
    
    // Convert to cluster format
    for (size_t i = 0; i < dims[0]; ++i) {
      larlite::larflow3dhit hit;
      hit[0] = flat_hits[i*3];
      hit[1] = flat_hits[i*3+1];
      hit[2] = flat_hits[i*3+2];
      cluster->push_back(hit);
    }
    
    H5Sclose(dataspace);
    H5Dclose(hits_dataset);
  }
  
  return cluster;
}

std::unique_ptr<larlite::pcaxis> NuVertexCandidateHDF5Reader::reconstructPCAxis(hid_t pca_group) {
  auto pca = std::make_unique<larlite::pcaxis>();
  
  std::vector<double> eigenvalues;
  readDoubleVector(pca_group, "eigenvalues", eigenvalues);
  
  hid_t eigenvec_dataset = H5Dopen2(pca_group, "eigenvectors", H5P_DEFAULT);
  if (eigenvec_dataset >= 0) {
    hid_t dataspace = H5Dget_space(eigenvec_dataset);
    hsize_t dims[2];
    H5Sget_simple_extent_dims(dataspace, dims, NULL);
    
    std::vector<float> flat_eigenvecs(dims[0] * dims[1]);
    H5Dread(eigenvec_dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, flat_eigenvecs.data());
    
    // Convert to pcaxis format
    std::vector<std::vector<double>> eigenvectors(3, std::vector<double>(3));
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        eigenvectors[i][j] = flat_eigenvecs[i * 3 + j];
      }
    }
    
    H5Sclose(dataspace);
    H5Dclose(eigenvec_dataset);
    
    // Set eigenvalues and eigenvectors
    // if (eigenvalues.size() >= 3) {
    //   double eigenvals[3] = {eigenvalues[0], eigenvalues[1], eigenvalues[2]};
    //   pca->set_eigenvalues(eigenvals);
    // }
    
    // pca->set_eigenvectors(eigenvectors);
  }
  
  // Read average position
  std::vector<double> avg_pos;
  readDoubleVector(pca_group, "average_position", avg_pos);
  if (avg_pos.size() >= 3) {
    double avg_position[3] = {avg_pos[0], avg_pos[1], avg_pos[2]};
    //pca->set_average_position(avg_position);
  }
  
  int num_hits = 0;
  readIntAttribute(pca_group, "num_hits", num_hits);
  //pca->set_num_hits_used(num_hits);
  
  return pca;
}

} // namespace util
} // namespace larflow