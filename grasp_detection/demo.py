import os
import argparse
import torch
import numpy as np
import open3d as o3d
from PIL import Image

from gsnet import AnyGrasp
from graspnetAPI import GraspGroup

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--max_gripper_width', type=float, default=0.1, help='Maximum gripper width (<=0.1m)')
parser.add_argument('--gripper_height', type=float, default=0.03, help='Gripper height')
parser.add_argument('--top_down_grasp', action='store_true', help='Output top-down grasps.')
parser.add_argument('--debug', action='store_true', help='Enable debug mode')
cfgs = parser.parse_args()
cfgs.max_gripper_width = max(0, min(0.1, cfgs.max_gripper_width))

def demo(data_dir):
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # GPU 0
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"Total VRAM: {info.total / 1024**2:.2f} MB")
    print(f"Used VRAM:  {info.used / 1024**2:.2f} MB")
    print(f"Free VRAM:  {info.free / 1024**2:.2f} MB")
    pynvml.nvmlShutdown()
    anygrasp = AnyGrasp(cfgs)
    anygrasp.load_net()

    # get data
    colors = np.array(Image.open(os.path.join(data_dir, 'color.png')), dtype=np.float32) / 255.0
    print(colors.shape)
    # depths = np.array(Image.open(os.path.join(data_dir, 'depth.png')))
    depths = np.load(os.path.join(data_dir, 'depths.npy'))
    print(depths.max(), depths.min())
    # get camera intrinsics
    fx, fy = 606.56, 606.72
    # cx, cy = 324, 236.26
    cx, cy = 236.26, 324
    scale = 1.0
    # fx, fy = 927.17, 927.37
    # cx, cy = 651.32, 349.62
    # scale = 1000.0

    # set workspace to filter output grasps
    xmin, xmax = -0.2, 0
    ymin, ymax = 0, 0.2
    zmin, zmax = 0.0, 1.0
    lims = [xmin, xmax, ymin, ymax, zmin, zmax]

    # get point cloud
    xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depths / scale
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    # set your workspace to crop point cloud
    mask = (points_z > 0) & (points_z < float('inf'))
    points = np.stack([points_x, points_y, points_z], axis=-1)
    points = points[mask].astype(np.float32)
    colors = colors[mask].astype(np.float32)
    print(points.min(axis=0), points.max(axis=0))

    import time
    start_time = time.time()

    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # GPU 0
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"Total VRAM: {info.total / 1024**2:.2f} MB")
    print(f"Used VRAM:  {info.used / 1024**2:.2f} MB")
    print(f"Free VRAM:  {info.free / 1024**2:.2f} MB")
    pynvml.nvmlShutdown()
    gg, cloud = anygrasp.get_grasp(points, colors, lims=lims, apply_object_mask=True, dense_grasp=False, collision_detection=True)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

    if len(gg) == 0:
        print('No Grasp detected after collision detection!')

    gg = gg.nms().sort_by_score()
    gg_pick = gg[0:20]
    print(gg_pick.scores)
    print("translation: ", gg_pick[0].translation)
    print("rotation_matrix: ", gg_pick[0].rotation_matrix)
    print("depth: ", gg_pick[0].depth)
    print("width: ", gg_pick[0].width)
    print('grasp score:', gg_pick[0].score)

    # visualization
    if cfgs.debug:
        # trans_mat = np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
        trans_mat = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
        cloud.transform(trans_mat)
        grippers = gg.to_open3d_geometry_list()
        for gripper in grippers:
            gripper.transform(trans_mat)
        # o3d.visualization.draw_geometries([*grippers, cloud])
        # o3d.visualization.draw_geometries([grippers[0], cloud])

        save_file = "grasp_all.png"

        visualizer = o3d.visualization.Visualizer()
        visualizer.create_window(visible=False)
        for gripper in grippers:
            visualizer.add_geometry(gripper)
        visualizer.add_geometry(cloud)
        visualizer.poll_events()
        visualizer.update_renderer()

        view_control = visualizer.get_view_control()
        zoom_scale_factor = 1.4
        view_control.scale(zoom_scale_factor)

        visualizer.capture_screen_image(save_file, do_render=True)
        print(f"Saved screen shot visualization at {save_file}")
        visualizer.destroy_window()

        save_file = "grasp_single.png"

        visualizer = o3d.visualization.Visualizer()
        visualizer.create_window(visible=False)
        gripper = grippers[0]
        visualizer.add_geometry(gripper)
        visualizer.add_geometry(cloud)
        visualizer.poll_events()
        visualizer.update_renderer()

        view_control = visualizer.get_view_control()
        zoom_scale_factor = 1.4
        view_control.scale(zoom_scale_factor)

        visualizer.capture_screen_image(save_file, do_render=True)
        print(f"Saved screen shot visualization at {save_file}")
        visualizer.destroy_window()


if __name__ == '__main__':
    
    demo('./example_data/')
