import torch
import torch.nn.functional as F

class DensityAwareSemanticSampling(torch.nn.Module):
    def __init__(self, n_samples, temperature=1.0 ):
        super().__init__()
        self.n_samples = n_samples
        self.temperature = temperature

    def compute_knearest_feature_density(self, points, features, kmax=16):
        """
        Compute features space density using kNN in positional space
        """
        Np, Nf = points.shape 
        distances = torch.cdist(points, points)     # position-space
        fdists    = torch.cdist(features,features)  # distances in feature-space
        # for each point, we get the k-nearest neighbors in position-space
        knn_dist,knn_indices = torch.topk(distances, k=min(kmax+1, distances.shape[-1]), dim=-1, largest=False)
        # above returns (values, indices) tensors and so [0] picks out only the values tensor
        # largest=True means that the smallest k values are returned
        # density = 1.0 / (torch.mean(knn_dist[..., 1:], dim=-1) + 1e-8)

        # print("knn_indices: ",knn_indices.shape)
        # print(knn_indices[:5,:])
        # print("pos-dists: ",distances.shape)
        # print(distances[:5,:])
        # print("fdists: ",fdists.shape)
        # print(fdists[:5,:])

        # gather the feature vectors
        kfeatures = torch.gather( fdists, 1, knn_indices[:,1:] )
        # print("kfeatures: ",kfeatures.shape)
        # print(kfeatures[:5,:])

        # gather the feature vectors
        density = torch.exp( -torch.mean(kfeatures, dim=-1)/self.temperature )
        return density, distances, fdists

    def forward(self, points, features):
        """
        Forward pass with cluster assignments
        
        Args:
            points: (N, 3) tensor of point coordinates
            features: (N, C) tensor of point features
            
        Returns:
            dict containing:
                sampled_points: (n_samples, 3) sampled points
                sampled_features: (n_samples, C) sampled features
                assignments: (N) cluster assignments for each point
                weights: (N) sampling weights used
        """
        N, _ = points.shape
        device = points.device

        # normalize features
        features_normalized = F.normalize(features, p=2, dim=-1)

        # Compute semantic distances, positional distances, and feature densitites
        density, pos_distance, semantic_distances = self.compute_knearest_feature_density(points, features_normalized)
        max_dist = torch.max(semantic_distances)
        #print("semantic_distances: ",semantic_distances.shape)
        #print(semantic_distances)

        # Compute point density
        max_density = torch.max(density)
        #density_normalized = density / density.max(dim=-1, keepdim=True)[0]
        density_row = density.unsqueeze(0).repeat(N,1) 
        density_col = density.unsqueeze(1).repeat(1,N)
        #print("density: ",density.shape)
        #print(density)
        #print(density_row)
        #print(density_col)
        # #dist_indicator = densitysq[:,:,:] > density[:,:]
        dist_indicator = density_row > density_col # element-wise
        #print(dist_indicator)
        ndist_above = torch.sum( dist_indicator, -1 ) # B, N
        #print("ndist_above: ",ndist_above)
        nonempty = ndist_above>0

        # get max dist for ndist_above==0 rows
        zero_semd = torch.max( semantic_distances[ ndist_above==0, : ], dim=-1 )[0]
        sdtemp = torch.clone( semantic_distances )
        dist_indicator[ndist_above==0,:] = True
        sdtemp[ dist_indicator==False ] = max_dist
        #print("semantic_distances after mod:")
        #print(sdtemp)
        delta_i = torch.min( sdtemp, dim=-1 )[0]
        #print("delta_i: ")
        #print(delta_i)

        delta_i[ ndist_above==0 ] = zero_semd
        #print("delta_i after max(dist) for empty density sets: ")
        #print(delta_i)

        sampling_score = density*delta_i
        #print("sampling_score: ")
        #print(sampling_score)

        kscores,kscores_indices = torch.topk(sampling_score, k=min(self.n_samples, sampling_score.shape[-1]), dim=-1, largest=True)
        #print("top-score indices: ",kscores_indices)

        # assign cluster index to these top indices
        top_dists = semantic_distances[ kscores_indices, : ]
        #print(top_dists)

        sampled_points   = points[ kscores_indices, : ]
        sampled_features = features[ kscores_indices, : ]
        assignments = torch.min( top_dists, dim=0 )[1] # return indices with [1]
        #print(assignments)
        
        return {
            'sampled_points': sampled_points,          # (n_samples, 3)
            'sampled_features': sampled_features,      # (n_samples, C)
            'assignments': assignments,                # (N)
            'weights': sampling_score                  # (N)
        }
