import torch

def fps_batchwise_fully_vectorized(points_batch, num_points):
    """
    Downsample a batch of point clouds using a fully vectorized batchwise FPS algorithm.

    Args:
        points_batch (torch.Tensor): The input batch of point clouds represented as a tensor of shape (batch, num_total_points, 3).
        num_points (int): The number of points to select for each point cloud.

    Returns:
        torch.Tensor: The downsampled batch of point clouds represented as a tensor of shape (batch, num_points, 3).
    """
    batch_size, num_total_points, _ = points_batch.shape

    # Create an identity matrix to help with excluding selected points
    identity_matrix = torch.eye(num_total_points, dtype=torch.bool).unsqueeze(0).expand(batch_size, -1, -1)

    # Initialize an array to store the selected point indices for each point cloud in the batch
    selected_indices_batch = []

    for _ in range(num_points):
        # Compute the distance from each point to all other points in each point cloud
        distances = torch.norm(points_batch.unsqueeze(2) - points_batch.unsqueeze(1), dim=-1)

        # Set distances to selected points to a large value to exclude them from selection
        distances[identity_matrix] = float('inf')

        # Find the farthest point in each point cloud
        farthest_indices = torch.argmin(distances, dim=2)

        # Append the selected indices for this iteration to the batch
        selected_indices_batch.append(farthest_indices)

        # Update the identity matrix to exclude the selected points in the next iteration
        identity_matrix.scatter_(2, farthest_indices.unsqueeze(2), False)

    # Concatenate the selected indices for each point cloud in the batch
    selected_indices_batch = torch.stack(selected_indices_batch, dim=2)

    # Create a new tensor containing only the selected points for each point cloud in the batch
    downsampled_points_batch = torch.gather(points_batch.unsqueeze(2).expand(-1, -1, num_points, -1), 1, selected_indices_batch.unsqueeze(-1).expand(-1, -1, -1, 3))

    return downsampled_points_batch

# Example usage:
if __name__ == "__main__":
    # Generate a random batch of point clouds for demonstration
    batch_size = 2
    num_total_points = 10000
    points_batch = torch.rand((batch_size, num_total_points, 3))  # Replace with your actual batch of point clouds
    
    # Set the desired number of points after downsampling for each point cloud
    num_points_after_downsampling = 1000
    
    # Perform batchwise FPS-based downsampling using fully vectorized code
    fps_downsampled_points_batch = fps_batchwise_fully_vectorized(points_batch, num_points_after_downsampling)
    
    # Save the FPS downsampled batch of point clouds or process it as needed
    print("FPS Downsampled Points Batch:", fps_downsampled_points_batch)