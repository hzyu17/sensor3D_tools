import lcm
from exlcm import pose_t
import open3d as o3d
import numpy as np
import rospy

import os, sys, time, threading

this_file = os.path.abspath(__file__)
this_dir  = os.path.dirname(this_file)
root_dir = os.path.dirname(this_dir)
scripts_dir = root_dir + "/scripts"
ros_dir = root_dir + "/ros"

if root_dir not in sys.path:            
    sys.path.insert(0, root_dir)


from scripts import OccpuancyGrid, SignedDistanceField3D
from scripts import SignedDistanceField
from ros import publish_voxel_grid

cell_size = 0.05
grid_dim = np.array([150, 150, 150], dtype=np.int32)
box_size = np.array([20, 20, 20], dtype=np.float64)


# rospy.init_node("voxel_to_rviz")

vis      = o3d.visualization.Visualizer()
vis.create_window(window_name="LCM Box Viewer")
box_mesh = None
lock     = threading.Lock()


def voxel_to_occmap(rows, cols, z, cell_size, voxel_grid):
    occmap = OccpuancyGrid(rows, cols, z, cell_size)
    occmap.from_voxel_grid(voxel_grid)
    return occmap


def create_box_mesh(center, size):
    box = o3d.geometry.TriangleMesh.create_box(
        width=size[0], height=size[1], depth=size[2]
    )
    # center the box around (0,0,0) before translation
    box.translate(-size/2.0)
    # now move to actual center
    box.translate(center)
    box.paint_uniform_color([1.0, 0.75, 0.5])
    return box


def pose_handler(channel, data):
    global box_mesh
    msg = pose_t.decode(data)
    T = np.asarray(msg.pose, dtype=np.float64).reshape(4, 4)
    center = T[:3, 3]

    # print("Received pcd message on channel \"%s\"" % channel)
    # print("   timestamp   = %s" % str(msg.timestamp))
    # # print("   pose    = %s" % str(msg.pose))
    # print("   center   = %s" % str(center))
    # print("")  

    occmap = OccpuancyGrid(*grid_dim, cell_size)
    occmap.add_obstacle(center, box_size)

    # Visualize the box in Open3D
    with lock:
        if box_mesh is None:
            box_mesh = create_box_mesh(center, box_size)
            vis.add_geometry(box_mesh)
        else:
            # compute how far we must move it:
            aabb = box_mesh.get_axis_aligned_bounding_box()
            # curr_center = aabb.get_center()
            # delta = center - np.asarray(curr_center)
            box_mesh.transform(T)
            # box_mesh.translate(delta)
            box_mesh.paint_uniform_color([0.2, 0.6, 0.9])
            
            vis.update_geometry(box_mesh)

    origin = np.array([
        occmap.origin_x,
        occmap.origin_y,
        occmap.origin_z
    ], dtype=np.float64)
    
    field3D = SignedDistanceField3D.generate_field3D(occmap.map.detach().numpy(), cell_size=cell_size)
    sdf = SignedDistanceField(origin, occmap.cell_size,
                              field3D.shape[0], field3D.shape[1], field3D.shape[2])
    for z in range(field3D.shape[2]):
        sdf.initFieldData(z, field3D[:,:,z])

    print("SDF Constructed!")

    # publish_voxel_grid(voxel_grid, topic="/obstacle_voxel", frame="world")
    # rospy.spin()
    
lc = lcm.LCM()
subscription = lc.subscribe("EXAMPLE", pose_handler)

# try:
#     while True:
#         lc.handle()
# except KeyboardInterrupt:
#     pass


def lcm_thread():
    while True:
        lc.handle()

# start LCM in background
threading.Thread(target=lcm_thread, daemon=True).start()

# main visualize loop (must run in main thread)
try:
    while True:
        with lock:
            vis.poll_events()      # process UI events
            vis.update_renderer()  # redraw
        time.sleep(0.02)          # ~=50 fps
except KeyboardInterrupt:
    pass

