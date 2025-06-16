#!/usr/bin/env python3
"""
Convert ROOT file with NuVertexCandidate objects to HDF5 format.

This script reads a ROOT file produced by KPSRecoManager containing
a tree with larflow::reco::NuVertexCandidate objects and converts
the data to HDF5 format for easier analysis.
"""

import ROOT
import h5py
import numpy as np
import argparse
import sys
from array import array


def get_vector_data(vec):
    """Convert ROOT vector to numpy array."""
    return np.array([vec[i] for i in range(vec.size())])


def get_lorentzvector_data(vec):
    """Convert ROOT TLorentzVector to numpy array [E, px, py, pz]."""
    return np.array([vec.E(), vec.Px(), vec.Py(), vec.Pz()])


def process_vtxcluster(cluster):
    """Convert VtxCluster_t to dictionary."""
    return {
        'producer': cluster.producer,
        'index': cluster.index,
        'dir': get_vector_data(cluster.dir),
        'pos': get_vector_data(cluster.pos),
        'gap': cluster.gap,
        'impact': cluster.impact,
        'npts': cluster.npts,
        'type': int(cluster.type)
    }


def process_track(track):
    """Convert larlite::track to dictionary with key properties."""
    # Extract key track properties
    n_points = track.NumberTrajectoryPoints()
    
    # Get trajectory points
    points = []
    for i in range(n_points):
        pt = track.LocationAtPoint(i)
        points.append([pt.X(), pt.Y(), pt.Z()])
    
    return {
        'id': track.ID(),
        'n_points': n_points,
        'points': np.array(points) if points else np.array([]),
        'length': track.Length()
    }


def process_larflowcluster(cluster):
    """Convert larlite::larflowcluster to dictionary."""
    # Get hit positions
    hits = []
    for i in range(cluster.size()):
        hit = cluster[i]
        hits.append([hit[0], hit[1], hit[2]])
    
    return {
        'n_hits': cluster.size(),
        'hits': np.array(hits) if hits else np.array([])
    }


def process_pcaxis(pca):
    """Convert larlite::pcaxis to dictionary."""
    return {
        'eigenvalues': np.array([pca.getEigenValues()[i] for i in range(3)]),
        'eigenvectors': np.array([[pca.getEigenVectors()[i][j] for j in range(3)] for i in range(3)]),
        'average_position': np.array([pca.getAvePosition()[i] for i in range(3)]),
        'num_hits': pca.getNumHitsUsed()
    }


def convert_nuvertex_candidate(vtx_candidate, entry_idx, vtx_idx, h5_group):
    """Convert a single NuVertexCandidate to HDF5 group."""
    # Create group for this vertex candidate
    vtx_name = f"entry_{entry_idx}_vtx_{vtx_idx}"
    vtx_group = h5_group.create_group(vtx_name)
    
    # Store basic vertex information
    vtx_group.attrs['keypoint_producer'] = vtx_candidate.keypoint_producer
    vtx_group.attrs['keypoint_index'] = vtx_candidate.keypoint_index
    vtx_group.attrs['keypoint_type'] = vtx_candidate.keypoint_type
    vtx_group.attrs['row'] = vtx_candidate.row
    vtx_group.attrs['tick'] = vtx_candidate.tick
    
    # Store position and column info
    vtx_group.create_dataset('pos', data=get_vector_data(vtx_candidate.pos))
    vtx_group.create_dataset('col_v', data=get_vector_data(vtx_candidate.col_v))
    
    # Store scores
    scores_group = vtx_group.create_group('scores')
    scores_group.attrs['score'] = vtx_candidate.score
    scores_group.attrs['maxScore'] = vtx_candidate.maxScore
    scores_group.attrs['avgScore'] = vtx_candidate.avgScore
    scores_group.attrs['netScore'] = vtx_candidate.netScore
    scores_group.attrs['netNuScore'] = vtx_candidate.netNuScore
    
    # Store clusters
    clusters_group = vtx_group.create_group('clusters')
    n_clusters = vtx_candidate.cluster_v.size()
    clusters_group.attrs['n_clusters'] = n_clusters
    
    for i in range(n_clusters):
        cluster_data = process_vtxcluster(vtx_candidate.cluster_v[i])
        cluster_grp = clusters_group.create_group(f'cluster_{i}')
        cluster_grp.attrs['producer'] = cluster_data['producer']
        cluster_grp.attrs['index'] = cluster_data['index']
        cluster_grp.attrs['gap'] = cluster_data['gap']
        cluster_grp.attrs['impact'] = cluster_data['impact']
        cluster_grp.attrs['npts'] = cluster_data['npts']
        cluster_grp.attrs['type'] = cluster_data['type']
        cluster_grp.create_dataset('dir', data=cluster_data['dir'])
        cluster_grp.create_dataset('pos', data=cluster_data['pos'])
    
    # Store cluster PCA
    cluster_pca_group = vtx_group.create_group('cluster_pca')
    n_pca = vtx_candidate.cluster_pca_v.size()
    cluster_pca_group.attrs['n_pca'] = n_pca
    
    for i in range(n_pca):
        pca_data = process_pcaxis(vtx_candidate.cluster_pca_v[i])
        pca_grp = cluster_pca_group.create_group(f'pca_{i}')
        pca_grp.create_dataset('eigenvalues', data=pca_data['eigenvalues'])
        pca_grp.create_dataset('eigenvectors', data=pca_data['eigenvectors'])
        pca_grp.create_dataset('average_position', data=pca_data['average_position'])
        pca_grp.attrs['num_hits'] = pca_data['num_hits']
    
    # Store tracks
    tracks_group = vtx_group.create_group('tracks')
    n_tracks = vtx_candidate.track_v.size()
    tracks_group.attrs['n_tracks'] = n_tracks
    
    # Track properties as arrays
    if n_tracks > 0:
        tracks_group.create_dataset('track_len_v', data=get_vector_data(vtx_candidate.track_len_v))
        #tracks_group.create_dataset('track_kemu_v', data=get_vector_data(vtx_candidate.track_kemu_v))
        #tracks_group.create_dataset('track_keproton_v', data=get_vector_data(vtx_candidate.track_keproton_v))
        #tracks_group.create_dataset('track_muid_v', data=get_vector_data(vtx_candidate.track_muid_v))
        #tracks_group.create_dataset('track_protonid_v', data=get_vector_data(vtx_candidate.track_protonid_v))
        #tracks_group.create_dataset('track_mu_vs_proton_llratio_v', 
        #                            data=get_vector_data(vtx_candidate.track_mu_vs_proton_llratio_v))
        tracks_group.create_dataset('track_isSecondary_v', data=get_vector_data(vtx_candidate.track_isSecondary_v))
        
        # Track directions (2D array)
        track_dirs = []
        for i in range(vtx_candidate.track_dir_v.size()):
            track_dirs.append(get_vector_data(vtx_candidate.track_dir_v[i]))
        tracks_group.create_dataset('track_dir_v', data=np.array(track_dirs))
        
        # Track momenta
        track_pmu = []
        track_pproton = []
        #for i in range(vtx_candidate.track_pmu_v.size()):
        #    track_pmu.append(get_lorentzvector_data(vtx_candidate.track_pmu_v[i]))
        #    track_pproton.append(get_lorentzvector_data(vtx_candidate.track_pproton_v[i]))
        #tracks_group.create_dataset('track_pmu_v', data=np.array(track_pmu))
        #tracks_group.create_dataset('track_pproton_v', data=np.array(track_pproton))
        
        # Individual track details
        for i in range(n_tracks):
            track_grp = tracks_group.create_group(f'track_{i}')
            track_data = process_track(vtx_candidate.track_v[i])
            track_grp.attrs['id'] = track_data['id']
            track_grp.attrs['n_points'] = track_data['n_points']
            track_grp.attrs['length'] = track_data['length']
            if track_data['points'].size > 0:
                track_grp.create_dataset('points', data=track_data['points'])
    
    # Store showers
    showers_group = vtx_group.create_group('showers')
    n_showers = vtx_candidate.shower_v.size()
    showers_group.attrs['n_showers'] = n_showers
    
    if n_showers > 0:
        showers_group.create_dataset('shower_isSecondary_v', data=get_vector_data(vtx_candidate.shower_isSecondary_v))
        
        # Shower plane properties (2D arrays)
        shower_pixsum = []
        shower_dqdx = []
        shower_mom = []
        
        for i in range(vtx_candidate.shower_plane_pixsum_vv.size()):
            shower_pixsum.append(get_vector_data(vtx_candidate.shower_plane_pixsum_vv[i]))
            shower_dqdx.append(get_vector_data(vtx_candidate.shower_plane_dqdx_vv[i]))
            
            # Momenta for this shower
            mom_planes = []
            for j in range(vtx_candidate.shower_plane_mom_vv[i].size()):
                mom_planes.append(get_lorentzvector_data(vtx_candidate.shower_plane_mom_vv[i][j]))
            shower_mom.append(mom_planes)
        
        showers_group.create_dataset('shower_plane_pixsum_vv', data=np.array(shower_pixsum))
        showers_group.create_dataset('shower_plane_dqdx_vv', data=np.array(shower_dqdx))
        # Note: shower_mom is a 3D array (shower x plane x 4-momentum)
        showers_group.create_dataset('shower_plane_mom_vv', data=np.array(shower_mom))
        
        # Individual shower details
        for i in range(n_showers):
            shower_grp = showers_group.create_group(f'shower_{i}')
            shower_data = process_larflowcluster(vtx_candidate.shower_v[i])
            shower_grp.attrs['n_hits'] = shower_data['n_hits']
            if shower_data['hits'].size > 0:
                shower_grp.create_dataset('hits', data=shower_data['hits'])
                
            # Shower trunk
            if i < vtx_candidate.shower_trunk_v.size():
                trunk_grp = shower_grp.create_group('trunk')
                trunk_data = process_track(vtx_candidate.shower_trunk_v[i])
                trunk_grp.attrs['id'] = trunk_data['id']
                trunk_grp.attrs['n_points'] = trunk_data['n_points']
                trunk_grp.attrs['length'] = trunk_data['length']
                if trunk_data['points'].size > 0:
                    trunk_grp.create_dataset('points', data=trunk_data['points'])
            
            # Shower PCA
            if i < vtx_candidate.shower_pcaxis_v.size():
                pca_data = process_pcaxis(vtx_candidate.shower_pcaxis_v[i])
                pca_grp = shower_grp.create_group('pcaxis')
                pca_grp.create_dataset('eigenvalues', data=pca_data['eigenvalues'])
                pca_grp.create_dataset('eigenvectors', data=pca_data['eigenvectors'])
                pca_grp.create_dataset('average_position', data=pca_data['average_position'])
                pca_grp.attrs['num_hits'] = pca_data['num_hits']


def convert_root_to_hdf5(input_file, output_file, tree_name="KPSRecoManagerTree", branch_name="nuvetoed_v"):
    """Main conversion function."""
    # Open ROOT file
    print(f"Opening ROOT file: {input_file}")
    root_file = ROOT.TFile.Open(input_file, "READ")
    if not root_file or root_file.IsZombie():
        print(f"Error: Cannot open ROOT file {input_file}")
        return False
    
    # Get tree
    tree = root_file.Get(tree_name)
    if not tree:
        print(f"Error: Cannot find tree '{tree_name}' in ROOT file")
        root_file.Close()
        return False
    
    print(f"Found tree '{tree_name}' with {tree.GetEntries()} entries")
    
    # Create HDF5 file
    print(f"Creating HDF5 file: {output_file}")
    with h5py.File(output_file, 'w') as h5_file:
        # Store metadata
        h5_file.attrs['source_file'] = input_file
        h5_file.attrs['tree_name'] = tree_name
        h5_file.attrs['branch_name'] = branch_name
        h5_file.attrs['n_entries'] = tree.GetEntries()
        
        # Create main group for vertex candidates
        vertices_group = h5_file.create_group('vertices')
        
        # Process each entry
        total_vertices = 0
        for entry_idx in range(tree.GetEntries()):
            if entry_idx % 10 == 0:
                print(f"Processing entry {entry_idx}/{tree.GetEntries()}")
            
            tree.GetEntry(entry_idx)
            
            # Get the vector of NuVertexCandidates
            nuvertex_v = getattr(tree, branch_name)
            n_vertices = nuvertex_v.size()
            
            # Store entry metadata
            entry_group = vertices_group.create_group(f'entry_{entry_idx}')
            entry_group.attrs['n_vertices'] = n_vertices
            
            # Process each vertex candidate
            for vtx_idx in range(n_vertices):
                vtx_candidate = nuvertex_v[vtx_idx]
                convert_nuvertex_candidate(vtx_candidate, entry_idx, vtx_idx, vertices_group)
                total_vertices += 1
        
        h5_file.attrs['total_vertices'] = total_vertices
        print(f"Converted {total_vertices} vertex candidates from {tree.GetEntries()} entries")
    
    root_file.Close()
    return True


def main():
    parser = argparse.ArgumentParser(description='Convert NuVertexCandidate ROOT file to HDF5')
    parser.add_argument('input_file', help='Input ROOT file path')
    parser.add_argument('output_file', help='Output HDF5 file path')
    parser.add_argument('--tree-name', default='KPSRecoManagerTree', help='Name of the ROOT tree (default: KPSRecoManagerTree)')
    parser.add_argument('--branch-name', default='nuvetoed_v', help='Name of the branch containing NuVertexCandidate vector (default: nuvetoed_v)')
    
    args = parser.parse_args()

    # This is not necessary because we have python bindings through ROOT
    # # Load ROOT libraries
    # print("Loading ROOT libraries...")
    # try:
    #     # Load necessary ROOT libraries
    #     ROOT.gSystem.Load("libBase")
    #     ROOT.gSystem.Load("libDataFormat")
    #     ROOT.gSystem.Load("libLArFlow_Reco")
    # except:
    #     print("Warning: Could not load some ROOT libraries. Make sure environment is properly set up.")
    
    # Perform conversion
    success = convert_root_to_hdf5(args.input_file, args.output_file, args.tree_name, args.branch_name)
    
    if success:
        print(f"Successfully converted {args.input_file} to {args.output_file}")
    else:
        print("Conversion failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
