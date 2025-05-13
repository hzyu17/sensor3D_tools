import torch
import open3d as o3d
import numpy as np
import json
from pathlib import Path


class OccpuancyGrid:
    def __init__(self, rows, cols, z, cell_size, origin=np.zeros(3)):
        self.rows = rows
        self.cols = cols
        self.z = z
        self.map = torch.zeros((rows, cols, z), dtype=torch.uint8)
        self.corner_idx = torch.empty((0, 6), dtype=torch.long)
        self.origin_x = origin[0]
        self.origin_y = origin[1]
        self.origin_z = origin[2]
        self.cell_size = cell_size     
                
    
    def set_origin(self, origin_x, origin_y, origin_z):
        """
        Set the origin of the occupancy grid in the world frame.
        """
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.origin_z = origin_z

    
    def add_obstacle(self, center, size):
        """
        Insert a rectangular obstacle into an occupancy grid and
        append its bounding box to the `corner` list.

        Parameters
        ----------
        center : (3,) Tensor-like [row, col, height] – grid-cell centre
        size     : (3,) Tensor-like [width, length, height] – side lengths (odd)

        Returns
        -------
        occ_map  : updated occupancy grid (same object, for convenience)
        corner   : updated (N+1, 6) tensor with bounding box [r0, r1, c0, c1, z0, z1]
        """
        
        # Convert matlab to python
        center = center - 1
        
        if center.ndim == 2:
            center = torch.concat((center, torch.tensor([size[2]/2])), dim=0)
        
        # Ensure integer tensors
        pos  = torch.as_tensor(center, dtype=torch.long)
        size = torch.as_tensor(size, dtype=torch.long)

        # Half-sizes 
        half = (size - 1) // 2

        # Inclusive index bounds in each dimension
        r0, r1 = (pos[0] - half[0]).item(), (pos[0] + half[0]).item()
        c0, c1 = (pos[1] - half[1]).item(), (pos[1] + half[1]).item()
        z0, z1 = (pos[2] - half[2]).item(), (pos[2] + half[2]).item()

        # Mark the cells as occupied (slice upper bound is exclusive in Python ⇒ +1)
        self.map[r0:r1 + 1, c0:c1 + 1, z0:z1 + 1] = 1

        # Append the axis-aligned bounding box to the running list
        new_box  = torch.tensor([r0, r1, c0, c1, z0, z1],
                                dtype=self.corner_idx.dtype, 
                                device=self.corner_idx.device).unsqueeze(0)
        
        self.corner_idx   = torch.cat((self.corner_idx, new_box), dim=0)
        
    
    def add_obstacle_idx(self, corner_idx):
        """
        corner_idx = [xmin xmax  ymin ymax  zmin zmax]  (inclusive voxel indices)
        """
        xmin, xmax, ymin, ymax, zmin, zmax = corner_idx
        # numpy slices are exclusive at the top ➜ +1
        self.map[ymin:ymax+1, xmin:xmax+1, zmin:zmax+1] = 1
        
    
    def corner_to_size(self, corner_idx):
        """
        corner_idx = [xmin xmax  ymin ymax  zmin zmax]  (inclusive voxel indices)
        """
        xmin, xmax, ymin, ymax, zmin, zmax = corner_idx.detach().numpy().tolist()
        # numpy slices are exclusive at the top ➜ +1
        return [xmax - xmin, ymax - ymin, zmax-zmin]
        
        
    @classmethod
    def from_json(cls, path):
        spec = json.loads(Path(path).read_text())
        omap = cls(spec["rows"], spec["cols"], spec["z"],
                   spec["cell_size"], spec["origin"])
        for obst in spec.get("obstacles", []):
            omap.add_obstacle_idx(obst["corner_idx"])
        return omap
    
    
    def save_to_json(self, json_file: str):
        """
        Serialize this occupancy grid to JSON with fields
        { cols, rows, z, origin, cell_size, obstacles:[{ corner_idx: [...] }, …] }.
        """
        meta = {
            "cols":      self.cols,
            "rows":      self.rows,
            "z":         self.z,
            "origin":    [self.origin_x, self.origin_y, self.origin_z],
            "cell_size": self.cell_size,
            # build a list of dicts for each obstacle
            "obstacles": [
                {"corner_idx": box.tolist(),
                 "size": self.corner_to_size(box)}
                for box in self.corner_idx
            ],
        }

        print("obstacles")
        print([
                {"corner_idx": box.tolist(),
                 "size": self.corner_to_size(box)}
                for box in self.corner_idx
            ])
        
        path = Path(json_file)
        path.write_text(json.dumps(meta, indent=2))
        
        print(f"Wrote {path}  ({len(meta['obstacles'])} obstacles)")
    

    def transform(self, pose, visualize=True):
        """
        Transform the occupancy grid by a given pose.
        """
        # 1) get all occupied voxel‑indices (N×3)
        idx = np.array(np.nonzero(self.map))  # shape (N,3)

        # 2) world‑coords, homogeneous
        pts = idx * self.cell_size + np.array([self.origin_x,
                                            self.origin_y,
                                            self.origin_z])
        pts_h = np.hstack([pts, np.ones((pts.shape[0],1))])  # (N,4)

        # 3) apply pose
        pts_t = (pose @ pts_h.T).T[:, :3]  # still shape (N,3)

        # 4) back into grid‑indices
        new_idx = np.round((pts_t - [self.origin_x,
                                    self.origin_y,
                                    self.origin_z])
                        / self.cell_size).astype(int)

        # 5) write into a fresh map
        new_map = torch.zeros_like(self.map)
        mask = np.all((new_idx >= 0) & 
                    (new_idx < np.array(self.map.shape)), axis=1)
        valid = new_idx[mask]
        new_map[valid[:,0], valid[:,1], valid[:,2]] = 1
        self.map = new_map

        if visualize:
            self.visualize()
        return

        
    
    def from_voxel_grid(self, voxel_grid):
        # Update the occupancies from a given voxel grid in open3d
        idx_xyz = np.asarray([v.grid_index for v in voxel_grid.get_voxels()], dtype=np.int64)
        
        self.clear()
        
        self.map[idx_xyz[:, 0], idx_xyz[:, 1], idx_xyz[:, 2]] = 1
        
        self.set_origin(voxel_grid.origin[0],
                        voxel_grid.origin[1],
                        voxel_grid.origin[2])
    
    
    def to_voxel_grid(self):
        """
        Convert this occupancy map (zeros = free, ones = occupied) into an
        Open3D VoxelGrid object.

        Returns
        -------
        o3d.geometry.VoxelGrid
            Sparse voxel grid whose `voxel_size` equals `self.cell_size`
            and whose `origin` equals the map's origin.
        """
        if not hasattr(self, "cell_size"):
            raise AttributeError("self must provide `cell_size` (linear voxel size).")
        if not hasattr(self, "origin_x"):
            raise AttributeError("self must provide `origin_x / y / z`.")

        # ------------------------------------------------------------------
        # 1. indices of occupied voxels (Nx, 3) in (x, y, z) order
        # ------------------------------------------------------------------
        print("Transforming occupancy map to voxel grid...")
        print("np.nonzero(self.map): ", np.nonzero(self.map))
        if len(np.nonzero(self.map)) == 0:
            raise RuntimeError("Map is empty!!")
        else:
            occ_idx = np.column_stack(np.nonzero(self.map))   # (N,3), each row = [ix, iy, iz]
            
        if occ_idx.size == 0:
            raise ValueError("Occupancy map contains no occupied cells.")

        # -----------------------------
        # 2. build sparse VoxelGrid
        # -----------------------------
        vg = o3d.geometry.VoxelGrid()
        vg.voxel_size = float(self.cell_size)
        vg.origin     = np.array([self.origin_x, self.origin_y, self.origin_z], dtype=np.float64)

        # create Voxel objects and attach them
        for ix, iy, iz in occ_idx.T:
            vg.add_voxel(o3d.geometry.Voxel(np.array([ix, iy, iz], dtype=np.int64)))
        return vg
    
    
    def origin(self):
        return np.array([self.origin_x, self.origin_y, self.origin_z])
        
    
    def clear(self):
        """
        Clear all occupancy grids.
        """
        self.map.zero_()
    
    
    def visualize(self):
        base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3) 
        vg = self.to_voxel_grid()

        print("Voxel origin: ", vg.origin)

        # # Visualize the occupancy grid with respect to the origin
        # import copy
        # vg_origin = copy.deepcopy(vg)
        # pose_origin = np.eye(4)
        # pose_origin[:3, 3] = np.array([self.origin_x, self.origin_y, self.origin_z])
        
        # # Convert the voxel grid to a point cloud
        # centers = []
        # colors  = []
        # for v in vg_origin.get_voxels():                 
        #     centre = vg_origin.get_voxel_center_coordinate(v.grid_index)
        #     centers.append(centre)
        #     if hasattr(v, "color"):               
        #         colors.append(v.color)
        # pc = o3d.geometry.PointCloud()
        # pc.points = o3d.utility.Vector3dVector(np.asarray(centers))
        
        # # Transform the point cloud using the given pose
        # pc.transform(pose_origin)
        
        # # Transform back the point cloud to a voxel grid
        # vg_origin = o3d.geometry.VoxelGrid.create_from_point_cloud(
        #     pc, voxel_size=vg_origin.voxel_size)
        
        
        vis = o3d.visualization.Visualizer()
        
        vis.create_window(window_name="Occupancy Map",
                        width=800, height=600)
        vis.add_geometry(vg)
        vis.add_geometry(base_frame)

        vis.run()
        vis.destroy_window()
    
    
if __name__ == '__main__':
    rows, cols, z = 100, 100, 100
    cell_size = 0.05
    occup_map = OccpuancyGrid(rows, cols, z, cell_size)   

    # --- Add a single rectangular obstacle ----------
    center = np.array([50, 50, 50])
    size   = np.array([50, 100, 50])

    occup_map.add_obstacle(center, size) 
    occup_map.visualize()