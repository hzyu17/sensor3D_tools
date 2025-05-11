import torch
import open3d as o3d
import numpy as np
import json

from pathlib import Path
import os, sys

this_file = os.path.abspath(__file__)
this_dir  = os.path.dirname(this_file)

if this_dir not in sys.path:    
    sys.path.insert(0, this_dir)

from OccupancyGrid import OccpuancyGrid
from PointCloud import PointCloud


def save_occmap_for_matlab(occup_map, filename):
    import scipy.io as sio
    class SimpleNamespace: pass
    dataset = SimpleNamespace()
    dataset.cols      = occup_map.cols
    dataset.rows      = occup_map.rows
    dataset.z         = occup_map.z
    dataset.origin_x  = occup_map.origin_x
    dataset.origin_y  = occup_map.origin_y
    dataset.origin_z  = occup_map.origin_z
    dataset.cell_size = occup_map.cell_size
    dataset.map = occup_map.map.detach().cpu().numpy().astype(np.float64)

    # --- pack into a dict, then save ------------------------------------
    mat_dict = {
        'cols'      : dataset.cols,
        'rows'      : dataset.rows,
        'z'         : dataset.z,
        'origin_x'  : dataset.origin_x,
        'origin_y'  : dataset.origin_y,
        'origin_z'  : dataset.origin_z,
        'cell_size' : dataset.cell_size,
        'map'       : dataset.map,
    }

    print("Saving the dataset to the mat file: ", filename)
    sio.savemat(filename, {'dataset': mat_dict}, do_compression=True)

    
    
def ply_to_mat(filename: str, mat_filename:str, cam2world:np.array, T_CO:np.array):
    pcd = PointCloud()

    # ----------------
    #  Read from file
    # ----------------
    pcd.read_from_file(this_dir+'/'+filename)

    pcd.register_camera_pose(torch.tensor(cam2world, dtype=torch.float32))
    pcd.update_obstacle_pose(torch.tensor(T_CO, dtype=torch.float32))
    pcd.draw_pcd_frame()

    # ----------- Convert point cloud to voxel and visualize -----------
    voxel_grid = pcd.to_voxel_grid(voxel_size=0.03)
    o3d.visualization.draw_geometries([voxel_grid, pcd.camera_frame, pcd.base_frame])
    center = voxel_grid.get_center()
    min_corner = voxel_grid.get_min_bound()
    max_corner = voxel_grid.get_max_bound()

    voxels = voxel_grid.get_voxels()

    print("Voxel grid center:", center)
    print("min corner:", min_corner)
    print("max corner:", max_corner)
    print("voxels:", voxels)

    # -------------------------------
    #  From voxels to occupancy grid
    # -------------------------------
    rows, cols, z = 100, 100, 100
    cell_size = 0.05
    occup_map = OccpuancyGrid(rows, cols, z, cell_size)
    occup_map.from_voxel_grid(voxel_grid)
    occup_map.set_origin(center[0], center[1], center[2])

    print("occupancy map shape:", occup_map.map.shape)

    save_occmap_for_matlab(occup_map, this_dir + '/' + mat_filename)
    
    
def transform_sdf(dataset_jsonfile, pose, visualize=True):
    """Transform a Signed Distance Function (SDF) according to a given pose, 
        and convert it to an voxel grid (occupancy map) as the return value.

    Args:
        dataset_jsonfile (str): json configuration file
        pose (np.array(4,4)): SE(3) pose to transform the sdf
        visualize (bool, optional): Defaults to True.

    Returns:
        vg_T: transformed voxel grid
        occup_map: transformed occupancy map
    """
    
    occup_map = OccpuancyGrid.from_json(dataset_jsonfile)
    vg_T = occup_map.transform(pose, visualize)
    
    return vg_T, occup_map
    
    

if __name__ == '__main__':
    
    # ==================
    #   Example Usage
    # ==================    
    # ---- Create an occupancy map with obstacles from json file, and transform using a pose -----
    json_file = this_dir+"/WAMDeskDataset.json"
    
    T_example = np.eye(4)
    T_example[:3, 3] = [0.0, 0.0, 0.0]
    T_example[:3, [0,1]] = T_example[:3, [1,0]]
    
    # Transform the sdf according to a given pose
    transformed_vg, transformed_occup_map = transform_sdf(json_file, T_example)
    
    # Convert the occupancy map to a sdf and save it to .bin file
    save_occmap_for_matlab(transformed_occup_map, this_dir+"/occupancy_map.mat")
    
    # read_and_save_sdf(this_dir+"/occupancy_map.mat", sdf_bin_filename)
    
    # --- Create an occupancy map with one obstacle -------
    rows, cols, z = 100, 100, 100
    cell_size = 0.05
    occup_map = OccpuancyGrid(rows, cols, z, cell_size)   

    # --- Add a single rectangular obstacle ----------
    center = (50, 50, 50)
    size   = (50, 100, 50)

    occup_map.add_obstacle(center, size) 

    # ----------------------
    #   Quick sanity check
    # ----------------------
    print("Number of occupied voxels:", int(occup_map.map.sum()))
    print("First obstacle AABB:", occup_map.corner_idx[0].tolist())

    pcd = PointCloud()
    # pcd.construct_pcd()


    # ----------------
    #  Read from file
    # ----------------
    pcd.read_from_file(this_dir+"/UTF-800000001.ply")

    # center_camera = torch.tensor([0.0, 0.0, 0.0])
    # size_camera   = torch.tensor([rows, cols, z])
    # cell_size = 0.05
    # pcd.to_occmap(center_camera, size_camera, cell_size)


    # Example of a camera pose
    R = np.array([[ 0, -1,  0],
                [ 1,  0,  0],
                [ 0,  0,  1]])
    t = np.array([0.1, 0.1, 0.5])

    # build the 4×4 matrix
    cam2world = np.eye(4)
    cam2world[:3, :3] = R
    cam2world[:3,  3] = t

    pcd.register_camera_pose(torch.tensor(cam2world, dtype=torch.float32))

    R_obj = np.array([[ 0, -1,  0],
                    [ 0,  0,  1],
                    [ 1,  0,  0]])
    t_obj = np.array([0, 0, 0])

    T_CO = np.eye(4)
    T_CO[:3, :3] = R_obj
    T_CO[:3,  3] = t_obj

    pcd.update_obstacle_pose(torch.tensor(T_CO, dtype=torch.float32))
    pcd.draw_pcd_frame()

    # ----------- Convert to voxel and visualize -----------
    voxel_grid = pcd.to_voxel_grid(voxel_size=0.03)
    o3d.visualization.draw_geometries([voxel_grid, pcd.camera_frame, pcd.base_frame])
    center = voxel_grid.get_center()
    min_corner = voxel_grid.get_min_bound()
    max_corner = voxel_grid.get_max_bound()

    voxels = voxel_grid.get_voxels()

    print("Voxel grid center:", center)
    print("min corner:", min_corner)
    print("max corner:", max_corner)
    print("voxels:", voxels)
    
    # -------------------------------
    #  From voxels to occupancy grid
    # -------------------------------
    rows, cols, z = 100, 100, 100
    cell_size = 0.05
    occup_map = OccpuancyGrid(rows, cols, z, cell_size)
    occup_map.from_voxel_grid(voxel_grid)
    

    print("occupancy map shape:", occup_map.map.shape)

    save_occmap_for_matlab(occup_map, this_dir+"/occupancy_map.mat")

    # ------------------------------------------------
    #   Transform the point cloud under a given pose
    # ------------------------------------------------

