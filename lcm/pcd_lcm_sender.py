import lcm
import time
import numpy as np
import os, sys

this_file = os.path.abspath(__file__)
this_dir  = os.path.dirname(this_file)
if this_dir not in sys.path:            
    sys.path.insert(0, this_dir)
    
from exlcm.pcd_t import pcd_t
from exlcm.pose_t import pose_t


def construct_pcdlcm_msg(pose, num_points, points, robot_base_offset):
    msg = pcd_t()
    msg.timestamp = int(time.time())
    msg.pose = pose
    msg.num_points = num_points
    msg.points = points
    msg.robot_base_offset = robot_base_offset

    return msg


def construct_poselcm_msg(pose, name='bodyframe'):
    msg = pose_t()
    msg.frame_name = name
    msg.timestamp = int(time.time())
    msg.pose = pose
    
    return msg


def publish_lcm_msg(msg, topic="EXAMPLE"):
    lc = lcm.LCM()
    lc.publish(topic, msg.encode())
    

if __name__ == '__main__':
    R_obj = np.array([[ 0, -1,  0],
                    [ 0,  0,  1],
                    [ 1,  0,  0]])
    t_obj = np.array([0, 0, 0])

    T_CO = np.eye(4, dtype=np.float32)
    T_CO[:3, :3] = R_obj
    T_CO[:3,  3] = t_obj
    
    num_points = 2048
    points = np.random.rand(num_points, 3).astype(np.float32)
    robot_base_offset = np.array([0, 0, 0], dtype=np.float32)
    
    msg = construct_pcdlcm_msg(T_CO, num_points, points, robot_base_offset)
    publish_lcm_msg(msg)
    
    