import torch
import open3d as o3d
import numpy as np
from OccupancyGrid import OccpuancyGrid



class PointCloud:
    def __init__(self, occup_map=None):
        self.occup_map = occup_map 
        
        self.base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3) 
        
        self.camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1) 
        
        # Default camera pose in the robot base frame
        self.T_BC = torch.eye(4, dtype=torch.float32)
        # Default obstacle pose in the camera frame
        self.T_CO = torch.eye(4, dtype=torch.float32)
        # Default obstacle pose in the robot base frame
        self.T_BO = self.T_BC @ self.T_CO
        
    
    def register_camera_pose(self, T_BC):
        self.T_BC = T_BC
        self.T_BO = self.T_BC @ self.T_CO
        self.camera_frame.transform(T_BC)
        
        self.pcd.transform(self.T_BO)
    
    
    def update_obstacle_pose(self, T_CO):
        """
        Update the obstacle pose of the occupancy grid in the camera frame.
        """
        self.T_CO = T_CO
        self.T_BO = self.T_BC @ self.T_CO
        
        self.pcd.transform(self.T_BO)
        
    
    def read_from_file(self, filename):
        self.pcd = o3d.io.read_point_cloud(filename)
        print("read pcd")
    
    def transform_frame(self, T):
        T[:3, 3] = [0.0, 0.0, 0.0]
        self.camera_frame = self.frame.transform(T.copy())   
    
    
    def construct_pcd(self):
        coords = torch.nonzero(self.occup_map.map).float()
        coords_metric = coords * self.occup_map.cell_size
        coords_metric[:,0] += self.occup_map.origin_x
        coords_metric[:,1] += self.occup_map.origin_y
        coords_metric[:,2] += self.occup_map.origin_z

        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(coords_metric.numpy())

    
    def draw_pcd_frame(self):
        self.pcd.paint_uniform_color([0.3, 0.4, 0.5])  
        self.pcd = self.pcd.voxel_down_sample(0.05)

        o3d.visualization.draw_geometries([self.pcd, 
                                           self.camera_frame, 
                                           self.base_frame],
                                            window_name="PCD + Camera Frame + Base Frame",
                                            width=800,
                                            height=600,
                                            point_show_normal=False)
        
        
    def to_occmap(self, center, size_xyz, cell_size):
        """Converting a point cloud into an occupancy map."""
        dims = (size_xyz / cell_size).long()   
        
        self.occ_map = OccpuancyGrid(*dims, cell_size)           
        # occ_map  = torch.zeros(dims.tolist(), dtype=torch.uint8)            

        # --- point cloud  -----------------------------------------------------
        pts = torch.from_numpy(np.asarray(self.pcd.points))    # (N,3) float32

        # --- world → grid -----------------------------------------------------
        grid_pts = ((pts - center) / cell_size).floor().long()  # (N,3) integer indices
        mask = ((grid_pts >= 0) & (grid_pts < dims)).all(dim=1)
        grid_pts = grid_pts[mask]                         # keep inside bounds
        
        self.occ_map.map[grid_pts] = 1
        
    def to_voxel_grid(self, voxel_size=0.01):
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(self.pcd, voxel_size=voxel_size)
        return voxel_grid