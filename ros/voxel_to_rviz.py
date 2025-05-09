import rospy, geometry_msgs.msg
import open3d as o3d
import numpy as np
from visualization_msgs.msg import Marker

from costmap_2d.msg import VoxelGrid as ROSVoxelGrid
from geometry_msgs.msg import Point32, Vector3, Pose
from std_msgs.msg import Header
from octomap_msgs.msg import Octomap, OctomapWithPose



def publish_voxel_grid(grid, topic="/voxel_map", frame="map"):
    pub = rospy.Publisher(topic, Marker, queue_size=1, latch=True)
    rospy.sleep(0.5)

    marker = Marker()
    marker.header.frame_id = frame
    marker.header.stamp = rospy.Time.now()
    marker.ns  = "voxels"
    marker.id  = 0
    marker.type = Marker.CUBE_LIST
    marker.action = Marker.ADD

    marker.scale.x = marker.scale.y = marker.scale.z = grid.voxel_size
    marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.2, 0.7, 1.0, 0.8

    origin = np.asarray(grid.origin)
    vs = grid.voxel_size
    for v in grid.get_voxels():
        centre = origin + (np.array(v.grid_index, float) + 0.5) * vs
        marker.points.append(geometry_msgs.msg.Point(*centre))

    pub.publish(marker)
    rospy.loginfo("Published %d voxels.", len(marker.points))
    
    
    
def o3d_voxelgrid_to_ros(o3d_grid: o3d.geometry.VoxelGrid,
                         frame_id: str = 'map') -> ROSVoxelGrid:
    # 1) get grid indices
    voxels  = o3d_grid.get_voxels()
    indices = np.stack([v.grid_index for v in voxels], axis=0)

    # 2) dims
    min_idx = indices.min(axis=0)
    max_idx = indices.max(axis=0)
    dims    = (max_idx - min_idx + 1).astype(int)
    size_x, size_y, size_z = dims.tolist()

    # 3) build flattened data array
    data = np.zeros(size_x*size_y*size_z, dtype=np.uint32)
    for idx in indices:
        ix, iy, iz = (idx - min_idx)
        lin        = ix + iy*size_x + iz*(size_x*size_y)
        data[lin]  = 100  # occupancy value

    # 4) world-frame origin
    min_bound = o3d_grid.get_min_bound()  # numpy.ndarray (3,) :contentReference[oaicite:3]{index=3}
    origin    = Point32(min_bound[0], min_bound[1], min_bound[2])

    # 5) fill ROS message
    msg = ROSVoxelGrid()
    msg.header     = Header(stamp=rospy.Time.now(), frame_id=frame_id)
    msg.origin     = origin
    msg.resolutions = Vector3(o3d_grid.voxel_size,
                              o3d_grid.voxel_size,
                              o3d_grid.voxel_size)
    msg.size_x     = size_x
    msg.size_y     = size_y
    msg.size_z     = size_z
    msg.data       = data.tolist()

    return msg

    
def rosvoxel_to_octomap(vox: ROSVoxelGrid) -> OctomapWithPose:
    
    octo = Octomap()
    octo.header   = vox.header
    octo.binary   = True                # assume occupancy only
    octo.id       = "OcTree"            # OctoMap’s internal tree type
    octo.resolution = vox.resolutions.x # assume cubic voxels
    # VoxelGrid stores 1 byte per cell; Octomap wants the raw bytes:
    octo.data     = vox.data            # as a bytes array
    
    owp = OctomapWithPose()
    owp.header = octo.header
    # fill in owp.origin if needed:
    owp.origin = Pose()  # identity
    owp.octomap = octo
    
    return owp