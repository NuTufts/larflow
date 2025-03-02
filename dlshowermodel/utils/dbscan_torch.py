import torch

def dbscan_torch(X, eps, min_samples):
    n_samples = X.shape[0]
    labels = torch.zeros(n_samples, dtype=torch.int)
    cluster_label = 0
    visited = torch.zeros(n_samples, dtype=torch.bool)

    for i in range(n_samples):
        if visited[i]:
            continue

        visited[i] = True
        neighbors = torch.norm(X - X[i], dim=1) < eps
        if neighbors.sum() < min_samples:
            labels[i] = -1  # Noise point
        else:
            cluster_label += 1
            labels[i] = cluster_label
            expand_cluster(X, labels, visited, neighbors, cluster_label, eps, min_samples)

    return labels

def expand_cluster(X, labels, visited, neighbors, cluster_label, eps, min_samples):
    queue = neighbors.nonzero().flatten().tolist()

    while queue:
        i = queue.pop(0)
        if not visited[i]:
            visited[i] = True
            new_neighbors = torch.norm(X - X[i], dim=1) < eps
            if new_neighbors.sum() >= min_samples:
                labels[i] = cluster_label
                queue.extend(new_neighbors.nonzero().flatten().tolist())
            else:
                labels[i] = -1  # Noise point