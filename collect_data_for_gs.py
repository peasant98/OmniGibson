"""
Example script demo'ing robot control.

Options for random actions, as well as selection of robot action space
"""
import json
import time
from PIL import Image
from scipy.spatial.transform import Rotation as R

import numpy as np
from pynput import keyboard
import matplotlib.pyplot as plt


import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
from omnigibson.robots import REGISTERED_ROBOTS
from omnigibson.robots.robot_base import BaseRobot
from omnigibson.utils.ui_utils import choose_from_options, KeyboardRobotController


CONTROL_MODES = dict(
    random="Use autonomous random actions (default)",
    teleop="Use keyboard control",
)

SCENES = dict(
    Rs_int="Realistic interactive home environment (default)",
    empty="Empty environment with no objects",
)

PANDA_MODEL = "FrankaPanda"
GS_IMAGE_PATH = "gs_images"
GS_DEPTH_PATH = "gs_depths"
GS_SEG_PATH = "gs_segs"

# Don't use GPU dynamics and use flatcache for performance boost
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True


def choose_controllers(robot, random_selection=False):
    """
    For a given robot, iterates over all components of the robot, and returns the requested controller type for each
    component.

    :param robot: BaseRobot, robot class from which to infer relevant valid controller options
    :param random_selection: bool, if the selection is random (for automatic demo execution). Default False

    :return dict: Mapping from individual robot component (e.g.: base, arm, etc.) to selected controller names
    """
    # Create new dict to store responses from user
    controller_choices = dict()

    # Grab the default controller config so we have the registry of all possible controller options
    default_config = robot._default_controller_config

    # Iterate over all components in robot
    for component, controller_options in default_config.items():
        # Select controller
        options = list(sorted(controller_options.keys()))
        choice = choose_from_options(
            options=options, name="{} controller".format(component), random_selection=random_selection
        )

        # Add to user responses
        controller_choices[component] = choice

    return controller_choices


x_pressed = False

def on_press(key):
    global x_pressed
    try:
        if key.char == 'x':
            x_pressed = True
    except AttributeError:
        pass

def on_release(key):
    global x_pressed
    try:
        if key.char == 'x':
            x_pressed = False
    except AttributeError:
        pass



def main(random_selection=False, headless=False, short_exec=False):
    """
    Robot control demo with selection
    Queries the user to select a robot, the controllers, a scene and a type of input (random actions or teleop)
    """
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Choose scene to load
    scene_model = "Rs_int"

    # use Franka Panda
    robot_name = PANDA_MODEL


    scene_cfg = dict()
    if scene_model == "empty":
        scene_cfg["type"] = "Scene"
    else:
        scene_cfg["type"] = "InteractiveTraversableScene"
        scene_cfg["scene_model"] = scene_model
        
    # Add the robot we want to load
    robot0_cfg = dict()
    robot0_cfg["type"] = robot_name
    robot0_cfg["obs_modalities"] = ["rgb", "depth", "seg_instance", "normal", "scan", "occupancy_grid", "seg_semantic"]
    
    robot0_cfg["action_type"] = "continuous"
    robot0_cfg["action_normalize"] = True

    # Compile config
    cfg = dict(scene=scene_cfg, robots=[robot0_cfg])
    
    # Create the environment
    env = og.Environment(configs=cfg)
    


    # Choose robot controller to use
    robot = env.robots[0]
    controller_choices = choose_controllers(robot=robot, random_selection=random_selection)

    # Choose control mode
    if random_selection:
        control_mode = "random"
    else:
        control_mode = choose_from_options(options=CONTROL_MODES, name="control mode")

    # Update the control mode of the robot
    controller_config = {component: {"name": name} for component, name in controller_choices.items()}
    robot.reload_controllers(controller_config=controller_config)

    # Because the controllers have been updated, we need to update the initial state so the correct controller state
    # is preserved
    env.scene.update_initial_state()

    # Update the simulator's viewer camera's pose so it points towards the robot
    og.sim.viewer_camera.set_position_orientation(
        position=np.array([1.46949, -3.97358, 2.21529]),
        orientation=np.array([0.56829048, 0.09569975, 0.13571846, 0.80589577]),
    )

    # Reset environment and robot
    env.reset()
    robot.reset()
    


    # Create teleop controller
    action_generator = KeyboardRobotController(robot=robot)
    

    # Register custom binding to reset the environment
    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.R,
        description="Reset the robot",
        callback_fn=lambda: env.reset(),
    )



    # Print out relevant keyboard info if using keyboard teleop
    if control_mode == "teleop":
        action_generator.print_keyboard_teleop_info()

    # Other helpful user info
    print("Running demo.")
    print("Press ESC to quit")
    
    
    
        


    # Loop control until user quits
    max_steps = -1 if not short_exec else 100
    step = 0
    env_robot: BaseRobot = env.robots[0]
    

    
    # sensor name is first key
    sensor_name = list(env_robot.sensors.keys())[0]
    
    instrinsic_matrix = env_robot.sensors[sensor_name].intrinsic_matrix
    
    
    transforms_json = {}
    
    transforms_json["w"] = 720
    transforms_json["h"] = 720
    
    transforms_json["fl_x"] = instrinsic_matrix[0][0]
    transforms_json["fl_y"] = instrinsic_matrix[1][1]
    
    transforms_json["cx"] = instrinsic_matrix[0][2]
    transforms_json["cy"] = instrinsic_matrix[1][2]
    
    transforms_json["k1"] = 0
    transforms_json["k2"] = 0
    transforms_json["p1"] = 0
    transforms_json["p2"] = 0
    
    transforms_json["camera_model"] = "OPENCV"
    
    transforms_json["frames"] = []
    
    
    image_idx = 0
    
    while step != max_steps:
        action = action_generator.get_random_action() if control_mode == "random" else action_generator.get_teleop_action()
        obs, reward, done, info = env.step(action=action)
        
        # if key is pressed, visualize the sensors and save the depths and rgb images and camera pose
        if x_pressed:
            robot_mask = env_robot.get_robot_mask()
            # save mask to GS_IMAGES
            mask_image = Image.fromarray((robot_mask * 255).astype(np.uint8))
            mask_path = f'{GS_IMAGE_PATH}/mask.png'
            mask_image.save(mask_path)
            
            sensor_frames = env_robot.get_vision_data()
            camera_pose = env_robot.sensors[sensor_name].get_position_orientation()
            
            # convert camera pose to matrix
            camera_pose_matrix = np.zeros((4, 4))
            
            # set position
            camera_pose_matrix[:3, 3] = camera_pose[0]
            
            x, y, z, w = camera_pose[1]
            
            quat = R.from_quat([x, y, z, w])
            rot = quat.as_matrix()
            
            camera_pose_matrix[:3, :3] = rot
            
            # set last element to 1
            camera_pose_matrix[3, 3] = 1
            
            # for now, we will focus on only depth and rgb images
            # size is (720, 720, 3) for rgb and (720, 720) for depth
            
            rgb = sensor_frames["rgb"]
            depth = sensor_frames["depth"]
            
            segmentation = sensor_frames["seg_semantic"]
            
            # we need to mask out the robot from all images
            rgb_image = Image.fromarray((rgb * 255).astype(np.uint8))
            rgb_path = f'{GS_IMAGE_PATH}/gs_img{image_idx}.png'
            rgb_image.save(rgb_path)
            
            depth_int = (depth * 1000).astype(np.uint16)
            depth_image = Image.fromarray(depth_int)
            depth_path = f'{GS_DEPTH_PATH}/gs_depth{image_idx}.png'
            depth_image.save(depth_path)
            
            segmentation_image = Image.fromarray((segmentation * 255).astype(np.uint8))
            seg_path = f'{GS_SEG_PATH}/gs_seg{image_idx}.png'
            segmentation_image.save(seg_path)
            
            image_idx += 1
            
            env_robot.visualize_sensors()
            
            new_frame = {}
            new_frame["file_path"] = rgb_path
            new_frame["depth_file_path"] = depth_path
            new_frame["seg_file_path"] = seg_path
            
            new_frame["transform_matrix"] = camera_pose_matrix.tolist()
            
            transforms_json["frames"].append(new_frame)
            
            filename = 'transforms.json'
            with open(filename, 'w') as json_file:
                json.dump(transforms_json, json_file, indent=4)
            
        step += 1

    # Always shut down the environment cleanly at the end
    env.close()


if __name__ == "__main__":
    main()
