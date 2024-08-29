# Non-Parametric Networks for 3D Point Cloud Classification
import torch
import torch.nn as nn
from pointnet2_ops import pointnet2_utils
import open3d as o3d
from .model_utils import *
import random
import math


def modified_fps_outlier_removal_torch(points, num_points):
    """
    Downsample a point cloud using modified Farthest Point Sampling with outlier removal.

    Args:
        points (torch.Tensor): The input point cloud represented as a tensor of shape (N, 3).
        num_points (int): The number of points to select.
        threshold (float): The distance threshold for outlier removal.

    Returns:
        torch.Tensor: The downsampled point cloud tensor of shape (num_points, 3).
    """
    num_total_points = points.shape[0]

    # Initialize an array to store the selected point indices
    selected_indices = []

    # Randomly select the first point index
    first_index = torch.randint(num_total_points, (1,))
    selected_indices.append(first_index.item())

    # Compute the distance from the first selected point to all other points
    distances = torch.norm(points - points[first_index], dim=1)

    # Iteratively select points that are farthest from the already selected points
    while len(selected_indices) < num_points:
        m_q3 = torch.quantile(distances, 0.75)
        m_q1 = torch.quantile(distances, 0.25)
        m_iqr = m_q3 - m_q1
        measurement = m_q3 + (1.5 * m_iqr)
        farthest_index = torch.argmax(distances)   
        # Check if the farthest point is an outlier based on the threshold
        if True:
          if distances[farthest_index] > measurement:
              value = torch.quantile(distances, 0.9)
              temp = distances - value
              temp[temp<0] = 100
              farthest_index = torch.argmin(temp)
        selected_indices.append(farthest_index.item())
        distances = torch.min(distances, torch.norm(points - points[farthest_index], dim=1))
    
    # Create a new tensor containing only the selected points
    downsampled_points = points[selected_indices]
    return torch.tensor(selected_indices)


# FPS + k-NN
class FPS_kNN(nn.Module):
    def __init__(self, group_num, k_neighbors):
        super().__init__()
        self.group_num = group_num
        self.k_neighbors = k_neighbors

    def forward(self, xyz, x):
        B, N, _ = xyz.shape

        # # grid sample
        # smp_points = []
        # test_point = o3d.geometry.PointCloud()
        # for i in range(B):
        #     tmp = xyz[i].permute(1,0)
        #     test_point.points, rs_idx = o3d.utility.voxel_down_sample_and_trace(xyz[i].cpu())
        #     smp_points.append(test_point.points)

        # # Random Sampling
        # rs_idx = torch.randperm(N)[:self.group_num].long().cuda().repeat(B, 1)
        # lc_xyz = index_points(xyz, rs_idx)
        # lc_x = index_points(x, rs_idx)

        # poission Sampling
        # sampler = PoissonDiskSampler(min_dist=0.1, max_attempts=30)
        # ps_idx = sampler.generate_poisson_disk_sample_indices(xyz, self.group_num).long()
        # lc_xyz = index_points(xyz, ps_idx)
        # lc_x = index_points(x, ps_idx)        
        
        # FPS
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.group_num).long()
        lc_xyz = index_points(xyz, fps_idx)
        lc_x = index_points(x, fps_idx)

        # FPS modified
        # fps_idx = torch.stack([modified_fps_outlier_removal_torch(xyz[b], self.group_num) for b in range(B)], dim=0)
        # lc_xyz = index_points(xyz, fps_idx)
        # lc_x = index_points(x, fps_idx)

        # kNN
        knn_idx = knn_point(self.k_neighbors, xyz, lc_xyz)
        knn_xyz = index_points(xyz, knn_idx)
        knn_x = index_points(x, knn_idx)

        return lc_xyz, lc_x, knn_xyz, knn_x


# Local Geometry Aggregation
class LGA(nn.Module):
    def __init__(self, out_dim, alpha, beta):
        super().__init__()
        self.geo_extract = PosE_Geo(3, out_dim, alpha, beta)

    def forward(self, lc_xyz, lc_x, knn_xyz, knn_x):

        # Normalize x (features) and xyz (coordinates)
        mean_x = lc_x.unsqueeze(dim=-2)
        std_x = torch.std(knn_x - mean_x)

        mean_xyz = lc_xyz.unsqueeze(dim=-2)
        std_xyz = torch.std(knn_xyz - mean_xyz)

        knn_x = (knn_x - mean_x) / (std_x + 1e-5)
        knn_xyz = (knn_xyz - mean_xyz) / (std_xyz + 1e-5)

        # Feature Expansion
        B, G, K, C = knn_x.shape
        knn_x = torch.cat([knn_x, lc_x.reshape(B, G, 1, -1).repeat(1, 1, K, 1)], dim=-1)

        # Geometry Extraction
        knn_xyz = knn_xyz.permute(0, 3, 1, 2)
        knn_x = knn_x.permute(0, 3, 1, 2)
        knn_x_w = self.geo_extract(knn_xyz, knn_x)

        return knn_x_w


# Pooling
class Pooling(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.out_transform = nn.Sequential(
                nn.BatchNorm1d(out_dim),
                nn.GELU())

    def forward(self, knn_x_w):
        # Feature Aggregation (Pooling)
        lc_x = knn_x_w.max(-1)[0] + knn_x_w.mean(-1)
        lc_x = self.out_transform(lc_x)
        return lc_x


# PosE for Raw-point Embedding 
class PosE_Initial(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, beta):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha, self.beta = alpha, beta

    def forward(self, xyz):
        B, _, N = xyz.shape    
        feat_dim = self.out_dim // (self.in_dim * 2)
        
        feat_range = torch.arange(feat_dim).float().cuda()     
        dim_embed = torch.pow(self.alpha, feat_range / feat_dim)
        div_embed = torch.div(self.beta * xyz.unsqueeze(-1), dim_embed)

        sin_embed = torch.sin(div_embed)
        cos_embed = torch.cos(div_embed)
        position_embed = torch.stack([sin_embed, cos_embed], dim=4).flatten(3)
        position_embed = position_embed.permute(0, 1, 3, 2).reshape(B, self.out_dim, N)
        
        return position_embed


# PosE for Local Geometry Extraction
class PosE_Geo(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, beta):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha, self.beta = alpha, beta
        
    def forward(self, knn_xyz, knn_x):
        B, _, G, K = knn_xyz.shape
        feat_dim = self.out_dim // (self.in_dim * 2)

        feat_range = torch.arange(feat_dim).float().cuda()     
        dim_embed = torch.pow(self.alpha, feat_range / feat_dim)
        div_embed = torch.div(self.beta * knn_xyz.unsqueeze(-1), dim_embed)

        sin_embed = torch.sin(div_embed)
        cos_embed = torch.cos(div_embed)
        position_embed = torch.stack([sin_embed, cos_embed], dim=5).flatten(4)
        position_embed = position_embed.permute(0, 1, 4, 2, 3).reshape(B, self.out_dim, G, K)

        # Weigh
        knn_x_w = knn_x + position_embed
        knn_x_w *= position_embed
        # no reweight
        # knn_x_w = knn_x
        return knn_x_w


# Non-Parametric Encoder
class EncNP(nn.Module):  
    def __init__(self, input_points, num_stages, embed_dim, k_neighbors, alpha, beta):
        super().__init__()
        self.input_points = input_points
        self.num_stages = num_stages
        self.embed_dim = embed_dim
        self.alpha, self.beta = alpha, beta

        # Raw-point Embedding
        self.raw_point_embed = PosE_Initial(3, self.embed_dim, self.alpha, self.beta)

        self.FPS_kNN_list = nn.ModuleList() # FPS, kNN
        self.LGA_list = nn.ModuleList() # Local Geometry Aggregation
        self.Pooling_list = nn.ModuleList() # Pooling
        
        out_dim = self.embed_dim
        group_num = self.input_points

        # Multi-stage Hierarchy
        for i in range(self.num_stages):
            out_dim = out_dim * 2
            group_num = group_num // 2
            self.FPS_kNN_list.append(FPS_kNN(group_num, k_neighbors))
            self.LGA_list.append(LGA(out_dim, self.alpha, self.beta))
            self.Pooling_list.append(Pooling(out_dim))


    def forward(self, xyz, x):

        # Raw-point Embedding
        x = self.raw_point_embed(x)

        # Multi-stage Hierarchy
        for i in range(self.num_stages):
            # FPS, kNN
            xyz, lc_x, knn_xyz, knn_x = self.FPS_kNN_list[i](xyz, x.permute(0, 2, 1))
            # Local Geometry Aggregation
            knn_x_w = self.LGA_list[i](xyz, lc_x, knn_xyz, knn_x)
            # Pooling
            x = self.Pooling_list[i](knn_x_w)

        # Global Pooling
        x = x.max(-1)[0] + x.mean(-1)

        return x


# Non-Parametric Network
class Point_NN(nn.Module):
    def __init__(self, input_points=1024, num_stages=4, embed_dim=72, k_neighbors=90, beta=1000, alpha=100):
        super().__init__()
        # Non-Parametric Encoder
        self.EncNP = EncNP(input_points, num_stages, embed_dim, k_neighbors, alpha, beta)


    def forward(self, x):
        # xyz: point coordinates
        # x: point features
        xyz = x.permute(0, 2, 1)

        # Non-Parametric Encoder
        x = self.EncNP(xyz, x)
        return x


class PoissonDiskSampler:
    def __init__(self, min_dist=0.01, max_attempts=30):
        self.min_dist = min_dist
        self.max_attempts = max_attempts

    def euclidean_distance(self, p1, p2):
        return torch.norm(p1 - p2)

    def generate_poisson_disk_sample_indices(self, point_cloud_batch, num_samples):
        batch_size = point_cloud_batch.size(0)
        sampled_indices_batch = []

        for i in range(batch_size):
            point_cloud = point_cloud_batch[i]
            sampled_indices = self._generate_sample_indices(point_cloud, num_samples)
            sampled_indices_batch.append(sampled_indices)

        return sampled_indices_batch

    def _generate_sample_indices(self, point_cloud, num_samples):
        sampled_indices = []

        # Create a grid to accelerate point selection
        cell_size = self.min_dist / math.sqrt(3)
        grid = {}

        def get_neighboring_points(point):
            x, y, z = point
            min_x, max_x = int((x - 2 * self.min_dist) / cell_size), int((x + 2 * self.min_dist) / cell_size)
            min_y, max_y = int((y - 2 * self.min_dist) / cell_size), int((y + 2 * self.min_dist) / cell_size)
            min_z, max_z = int((z - 2 * self.min_dist) / cell_size), int((z + 2 * self.min_dist) / cell_size)

            neighboring_points = []

            for i in range(min_x, max_x + 1):
                for j in range(min_y, max_y + 1):
                    for k in range(min_z, max_z + 1):
                        if (i, j, k) in grid:
                            neighboring_points.extend(grid[(i, j, k)])

            return neighboring_points

        def is_valid_point(new_point):
            for existing_point,idx in get_neighboring_points(new_point):
                if self.euclidean_distance(existing_point, new_point) < self.min_dist:
                    return False
            return True

        def add_point_to_grid(point, index):
            grid_key = tuple(int(coord / cell_size) for coord in point)
            if grid_key not in grid:
                grid[grid_key] = []
            grid[grid_key].append((point, index))

        # Randomly select the first point from the point cloud
        first_point_index = random.randint(0, point_cloud.size(0) - 1)
        first_point = point_cloud[first_point_index]
        sampled_indices.append(first_point_index)
        
        sampled_count = 1  # Initialize sampled count
        while sampled_count < num_samples:
            print(sampled_count)
            current_index = random.randint(0, len(sampled_indices) -1) if sampled_count > 1 else 0
            current_point = point_cloud[sampled_indices[current_index]]
            found_valid_point = False

            for _ in range(self.max_attempts):
                theta = random.uniform(0, 2 * math.pi)
                phi = random.uniform(0, math.pi)
                r = random.uniform(self.min_dist, 2 * self.min_dist)

                x = current_point[0] + r * math.sin(phi) * math.cos(theta)
                y = current_point[1] + r * math.sin(phi) * math.sin(theta)
                z = current_point[2] + r * math.cos(phi)

                new_point = torch.tensor([x, y, z]).cuda()

                if (
                    0 <= x < point_cloud[:, 0].max() and
                    0 <= y < point_cloud[:, 1].max() and
                    0 <= z < point_cloud[:, 2].max() and
                    is_valid_point(new_point)
                ):
                    closest_index = torch.argmin(torch.norm(point_cloud - new_point, dim=1))
                    sampled_indices.append(closest_index)
                    add_point_to_grid(new_point, closest_index)
                    if sampled_count == 1:
                        add_point_to_grid(first_point, first_point_index)
                    sampled_count += 1
                    found_valid_point = True
                    break

            if not found_valid_point:
                # Remove the point from consideration if no valid point is found
                sampled_indices.pop(current_index)
                if len(sampled_indices) == 0:
                    # Randomly select the first point from the point cloud
                    first_point_index = random.randint(0, point_cloud.size(0) - 1)
                    first_point = point_cloud[first_point_index]
                    sampled_indices.append(first_point_index)
                    # add_point_to_grid(first_point, first_point_index)                

        return sampled_indices
        
       
