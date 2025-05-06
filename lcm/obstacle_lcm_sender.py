import time, math
import numpy as np

from pcd_lcm_sender import construct_poselcm_msg, publish_lcm_msg

R_obj = np.array([[ 0, -1,  0],
                [ 0,  0,  1],
                [ 1,  0,  0]])

T_CO = np.eye(4, dtype=np.float32)
T_CO[:3, :3] = R_obj

radius = 5.0
center_base = np.array([10.0, 10.0, 25.0], dtype=np.float32)
period = 50

for i in range(period):
    θ = 2*math.pi * i / period
    x = center_base[0] + radius * math.cos(θ)
    y = center_base[1] + radius * math.sin(θ)
    z = center_base[2]   # keep Z fixed

    T_CO[:3, 3] = np.array([x, y, z], dtype=np.float32)

    msg = construct_poselcm_msg(T_CO)
    publish_lcm_msg(msg)

    print(f"Sent circular pose #{i}: {[x, y, z]}")
    time.sleep(0.5)
