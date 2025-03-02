import torch
import torch.nn.functional as F

class SemanticDistanceSampling(torch.nn.Module):
    def __init__(self, n_samples, temperature=0.1):
        super().__init__()
        self.n_samples = n_samples
        self.temperature = temperature

    def forward(self, points, features):
        """
        Semantic distance-based point sampling with assignment tracking
        
        Args:
            points: (B, N, 3) tensor of point coordinates
            features: (B, N, C) tensor of point features
            
        Returns:
            sampled_points: (B, n_samples, 3) tensor of sampled points
            sampled_features: (B, n_samples, C) tensor of sampled features
            sample_indices: (B, n_samples) indices of selected points
            assignments: (B, N) tensor mapping each original point to nearest sampled point
            assignment_distances: (B, N) semantic distances to assigned centroids
        """
        B, N, _ = points.shape
        device = points.device

        # Normalize features
        features_normalized = F.normalize(features, p=2, dim=-1)
        
        # Compute pairwise semantic distances
        distances = torch.cdist(features_normalized, features_normalized)
        
        # Initialize first point randomly
        first_idx = torch.randint(0, N, (B,), device=device)
        indices = first_idx.unsqueeze(-1)
        
        # Initialize distance mask
        mask = torch.ones((B, N), device=device)
        mask[torch.arange(B, device=device), first_idx] = 0

        # Iteratively sample points
        for i in range(1, self.n_samples):
            # Get distances to already sampled points
            dist_to_selected = distances[
                torch.arange(B, device=device).unsqueeze(-1),
                indices,
                :
            ]  # [B, i, N]
            
            # Minimum distance to any selected point
            min_dist = dist_to_selected.min(dim=1)[0]  # [B, N]
            
            # Apply temperature and masking
            scores = torch.exp(min_dist / self.temperature) * mask
            probs = scores / scores.sum(dim=-1, keepdim=True)
            
            # Sample next point
            next_idx = torch.multinomial(probs, 1)
            indices = torch.cat([indices, next_idx], dim=-1)
            
            # Update mask
            mask[torch.arange(B, device=device), next_idx.squeeze(-1)] = 0

        # Compute assignments for all points
        sampled_features = torch.gather(
            features,
            1,
            indices.unsqueeze(-1).expand(-1, -1, features.shape[-1])
        )
        
        # Compute distances to sampled points
        assignment_distances = torch.cdist(
            features_normalized,
            F.normalize(sampled_features, p=2, dim=-1)
        )  # [B, N, n_samples]
        
        # Get assignments (indices of nearest sampled point)
        assignments = torch.argmin(assignment_distances, dim=2)  # [B, N]
        min_distances = torch.min(assignment_distances, dim=2)[0]  # [B, N]
        
        # Gather sampled points
        sampled_points = torch.gather(
            points,
            1,
            indices.unsqueeze(-1).expand(-1, -1, points.shape[-1])
        )

        return {
            'sampled_points': sampled_points,          # [B, n_samples, 3]
            'sampled_features': sampled_features,      # [B, n_samples, C]
            'sample_indices': indices,                 # [B, n_samples]
            'assignments': assignments,                # [B, N]
            'assignment_distances': min_distances      # [B, N]
        }