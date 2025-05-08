"""
Point cloud processing utilities for 3D scanning pipeline
Provides standalone functions for point cloud manipulation and processing
"""
import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np

from ..utils.exceptions import ProcessingError

logger = logging.getLogger(__name__)

class PointCloudProcessor:
    """
    Utility class for processing 3D point cloud data
    Provides methods for cleaning, transforming, and analyzing point clouds
    """
    
    @staticmethod
    def filter_outliers(points: np.ndarray, 
                       std_ratio: float = 2.0) -> np.ndarray:
        """
        Remove statistical outliers from point cloud
        
        Args:
            points: Input point cloud array
            std_ratio: Standard deviation threshold for outlier removal
            
        Returns:
            np.ndarray: Filtered point cloud with outliers removed
        """
        logger.info(f"Filtering outliers from {len(points)} points (std_ratio={std_ratio})")
        
        # Calculate mean distance to nearest neighbors for each point
        distances = np.zeros(len(points))
        for i in range(len(points)):
            # Use only the XYZ coordinates (first 3 columns)
            distances[i] = np.mean(np.linalg.norm(
                points[:, 0:3] - points[i, 0:3], axis=1
            ))
        
        # Filter based on statistical outlier removal
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        threshold = mean_dist + std_ratio * std_dist
        mask = distances < threshold
        
        filtered_points = points[mask]
        num_removed = len(points) - len(filtered_points)
        
        logger.info(f"Removed {num_removed} outlier points ({num_removed/len(points)*100:.1f}%)")
        
        return filtered_points
    
    @staticmethod
    def voxel_downsample(points: np.ndarray, 
                        voxel_size: float) -> np.ndarray:
        """
        Downsample point cloud using voxel grid
        
        Args:
            points: Input point cloud array
            voxel_size: Size of voxel grid for downsampling
            
        Returns:
            np.ndarray: Downsampled point cloud
        """
        logger.info(f"Voxel downsampling {len(points)} points (voxel_size={voxel_size})")
        
        # In a real implementation, this would use a proper voxel grid algorithm
        # For simulation purposes, we'll implement a simple version:
        
        # Get the bound of points
        min_bound = np.min(points[:, 0:3], axis=0)
        max_bound = np.max(points[:, 0:3], axis=0)
        
        # Compute voxel grid dimensions
        dim = (max_bound - min_bound) / voxel_size
        dim = np.ceil(dim).astype(int)
        
        # Compute voxel index for each point
        voxel_indices = np.floor((points[:, 0:3] - min_bound) / voxel_size).astype(int)
        
        # Convert to 1D index for grouping
        voxel_hash = voxel_indices[:, 0] + voxel_indices[:, 1] * dim[0] + voxel_indices[:, 2] * dim[0] * dim[1]
        
        # Group points by voxel
        unique_voxels, inverse_indices, counts = np.unique(voxel_hash, return_inverse=True, return_counts=True)
        
        # For each voxel, find the point closest to center
        downsampled_points = []
        for i, voxel_id in enumerate(unique_voxels):
            # Get points in this voxel
            voxel_mask = voxel_hash == voxel_id
            voxel_points = points[voxel_mask]
            
            # Compute centroid of voxel
            centroid = np.mean(voxel_points[:, 0:3], axis=0)
            
            # Find point closest to centroid
            distances = np.linalg.norm(voxel_points[:, 0:3] - centroid, axis=1)
            closest_point_idx = np.argmin(distances)
            
            downsampled_points.append(voxel_points[closest_point_idx])
        
        result = np.array(downsampled_points)
        
        logger.info(f"Downsampled to {len(result)} points ({len(result)/len(points)*100:.1f}% of original)")
        
        return result
    
    @staticmethod
    def estimate_normals(points: np.ndarray, 
                        k_neighbors: int = 30) -> np.ndarray:
        """
        Estimate normal vectors for each point in the cloud
        
        Args:
            points: Input point cloud array (XYZ coordinates)
            k_neighbors: Number of neighbors to use for normal estimation
            
        Returns:
            np.ndarray: Normal vectors for each point
        """
        logger.info(f"Estimating normals for {len(points)} points (k={k_neighbors})")
        
        # In a real implementation, this would use efficient nearest neighbor search
        # and proper normal estimation algorithms (e.g., from Open3D)
        
        # For simulation, we'll generate plausible normals for an ear canal
        # Assume the ear canal is roughly along the z-axis with variable radius
        
        # Extract XYZ coordinates
        coords = points[:, 0:3] if points.shape[1] >= 3 else points
        
        # For ear canal, approximate the centerline
        # Assuming z is depth into ear, x-y are cross-section
        center_x = np.mean(coords[:, 0])
        center_y = np.mean(coords[:, 1])
        
        # Estimate normals as pointing inward from the surface toward the centerline
        normals = np.zeros((len(coords), 3))
        
        for i in range(len(coords)):
            # Vector from center to point (in x-y plane)
            direction = np.array([coords[i, 0] - center_x, coords[i, 1] - center_y, 0])
            
            # Normalize to unit vector 
            norm = np.linalg.norm(direction)
            if norm > 1e-10:  # Avoid division by zero
                direction = direction / norm
            else:
                direction = np.array([0, 0, 1])  # Default for center points
            
            # Add some depth-dependent tilt based on z-coordinate
            z_factor = coords[i, 2] / np.max(coords[:, 2])
            tilt = np.array([0, 0, 0.2 + 0.3 * z_factor])
            
            # Combine for final normal vector
            normal = -direction + tilt  # Negative direction points inward
            normal = normal / np.linalg.norm(normal)  # Normalize
            
            normals[i] = normal
        
        # Add some random variation to make it more realistic
        normals += np.random.normal(0, 0.1, normals.shape)
        
        # Re-normalize to unit vectors
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0  # Avoid division by zero
        normals = normals / norms
        
        logger.info(f"Normal estimation complete")
        
        return normals
    
    @staticmethod
    def crop_region_of_interest(points: np.ndarray, 
                               min_bound: np.ndarray, 
                               max_bound: np.ndarray) -> np.ndarray:
        """
        Crop point cloud to region of interest defined by bounds
        
        Args:
            points: Input point cloud array
            min_bound: Minimum XYZ coordinates for bounding box
            max_bound: Maximum XYZ coordinates for bounding box
            
        Returns:
            np.ndarray: Cropped point cloud within bounds
        """
        logger.info(f"Cropping points to region of interest")
        
        # Extract XYZ coordinates
        coords = points[:, 0:3] if points.shape[1] >= 3 else points
        
        # Create mask for points within bounds
        mask = np.all((coords >= min_bound) & (coords <= max_bound), axis=1)
        
        # Apply mask to original points (preserving all columns)
        cropped_points = points[mask]
        
        logger.info(f"Cropped from {len(points)} to {len(cropped_points)} points")
        
        return cropped_points
    
    @staticmethod
    def align_to_canonical_orientation(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align point cloud to canonical orientation using PCA
        
        Args:
            points: Input point cloud array
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Transformed points and transformation matrix
        """
        logger.info(f"Aligning {len(points)} points to canonical orientation")
        
        # Extract XYZ coordinates
        coords = points[:, 0:3] if points.shape[1] >= 3 else points
        
        # Center the points
        centroid = np.mean(coords, axis=0)
        centered = coords - centroid
        
        # Compute covariance matrix and its eigenvectors/values
        cov = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Sort eigenvectors by eigenvalues in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Create rotation matrix from eigenvectors
        # We want to ensure a right-handed coordinate system
        if np.linalg.det(eigenvectors) < 0:
            eigenvectors[:, 2] *= -1
        
        # Create full transformation matrix (rotation + translation)
        transform = np.eye(4)
        transform[0:3, 0:3] = eigenvectors
        transform[0:3, 3] = -centroid
        
        # Apply transformation to original points
        homogeneous = np.ones((len(coords), 4))
        homogeneous[:, 0:3] = coords
        
        transformed_points = np.dot(homogeneous, transform.T)[:, 0:3]
        
        # Reconstruct full point cloud with all columns
        result = points.copy()
        result[:, 0:3] = transformed_points
        
        logger.info(f"Alignment complete")
        
        return result, transform
    
    @staticmethod
    def compute_point_cloud_features(points: np.ndarray) -> Dict[str, Any]:
        """
        Compute various statistical features of a point cloud
        
        Args:
            points: Input point cloud array
            
        Returns:
            Dict[str, Any]: Dictionary of computed features
        """
        logger.info(f"Computing statistical features for {len(points)} points")
        
        # Extract XYZ coordinates
        coords = points[:, 0:3] if points.shape[1] >= 3 else points
        
        # Compute basic statistics
        bounds_min = np.min(coords, axis=0)
        bounds_max = np.max(coords, axis=0)
        dimensions = bounds_max - bounds_min
        centroid = np.mean(coords, axis=0)
        
        # Compute density and spacing
        volume = np.prod(dimensions)
        density = len(coords) / volume if volume > 0 else 0
        
        # Compute average distance between points (approximate)
        avg_spacing = (volume / len(coords)) ** (1/3) if len(coords) > 0 else 0
        
        # Compute PCA statistics
        centered = coords - centroid
        cov = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Sort eigenvectors by eigenvalues in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        
        # Compute normalized eigenvalues for shape analysis
        normalized_eigenvalues = eigenvalues / np.sum(eigenvalues) if np.sum(eigenvalues) > 0 else np.zeros(3)
        
        # Compute shape factors
        linearity = (eigenvalues[0] - eigenvalues[1]) / eigenvalues[0] if eigenvalues[0] > 0 else 0
        planarity = (eigenvalues[1] - eigenvalues[2]) / eigenvalues[0] if eigenvalues[0] > 0 else 0
        sphericity = eigenvalues[2] / eigenvalues[0] if eigenvalues[0] > 0 else 0
        
        features = {
            "num_points": len(points),
            "bounds_min": bounds_min,
            "bounds_max": bounds_max,
            "dimensions": dimensions,
            "volume": volume,
            "centroid": centroid,
            "point_density": density,
            "average_spacing": avg_spacing,
            "eigenvalues": eigenvalues,
            "normalized_eigenvalues": normalized_eigenvalues,
            "linearity": linearity,
            "planarity": planarity,
            "sphericity": sphericity
        }
        
        logger.info(f"Feature computation complete")
        
        return features
