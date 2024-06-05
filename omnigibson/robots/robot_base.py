from abc import abstractmethod
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

from omnigibson.macros import create_module_macros
from omnigibson.sensors import create_sensor, SENSOR_PRIMS_TO_SENSOR_CLS, ALL_SENSOR_MODALITIES, VisionSensor, ScanSensor
from omnigibson.objects.usd_object import USDObject
from omnigibson.objects.object_base import BaseObject
from omnigibson.objects.controllable_object import ControllableObject
from omnigibson.utils.gym_utils import GymObservable
from omnigibson.utils.usd_utils import add_asset_to_stage
from omnigibson.utils.python_utils import classproperty, merge_nested_dicts
from omnigibson.utils.vision_utils import segmentation_to_rgb
from omnigibson.utils.constants import PrimType

# Global dicts that will contain mappings
REGISTERED_ROBOTS = dict()

# Add proprio sensor modality to ALL_SENSOR_MODALITIES
ALL_SENSOR_MODALITIES.add("proprio")

# Create settings for this module
m = create_module_macros(module_path=__file__)

# Name of the category to assign to all robots
m.ROBOT_CATEGORY = "agent"


class BaseRobot(USDObject, ControllableObject, GymObservable):
    """
    Base class for USD-based robot agents.

    This class handles object loading, and provides method interfaces that should be
    implemented by subclassed robots.
    """
    def __init__(
        self,
        # Shared kwargs in hierarchy
        name,
        prim_path=None,
        uuid=None,
        scale=None,
        visible=True,
        fixed_base=False,
        visual_only=False,
        self_collisions=False,
        load_config=None,

        # Unique to USDObject hierarchy
        abilities=None,

        # Unique to ControllableObject hierarchy
        control_freq=None,
        controller_config=None,
        action_type="continuous",
        action_normalize=True,
        reset_joint_pos=None,

        # Unique to this class
        obs_modalities="all",
        proprio_obs="default",
        sensor_config=None,

        **kwargs,
    ):
        """
        Args:
            name (str): Name for the object. Names need to be unique per scene
            prim_path (None or str): global path in the stage to this object. If not specified, will automatically be
                created at /World/<name>
            uuid (None or int): Unique unsigned-integer identifier to assign to this object (max 8-numbers).
                If None is specified, then it will be auto-generated
            scale (None or float or 3-array): if specified, sets either the uniform (float) or x,y,z (3-array) scale
                for this object. A single number corresponds to uniform scaling along the x,y,z axes, whereas a
                3-array specifies per-axis scaling.
            visible (bool): whether to render this object or not in the stage
            fixed_base (bool): whether to fix the base of this object or not
            visual_only (bool): Whether this object should be visual only (and not collide with any other objects)
            self_collisions (bool): Whether to enable self collisions for this object
            load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
                loading this prim at runtime.
            abilities (None or dict): If specified, manually adds specific object states to this object. It should be
                a dict in the form of {ability: {param: value}} containing object abilities and parameters to pass to
                the object state instance constructor.
            control_freq (float): control frequency (in Hz) at which to control the object. If set to be None,
                simulator.import_object will automatically set the control frequency to be at the render frequency by default.
            controller_config (None or dict): nested dictionary mapping controller name(s) to specific controller
                configurations for this object. This will override any default values specified by this class.
            action_type (str): one of {discrete, continuous} - what type of action space to use
            action_normalize (bool): whether to normalize inputted actions. This will override any default values
                specified by this class.
            reset_joint_pos (None or n-array): if specified, should be the joint positions that the object should
                be set to during a reset. If None (default), self._default_joint_pos will be used instead.
                Note that _default_joint_pos are hardcoded & precomputed, and thus should not be modified by the user.
                Set this value instead if you want to initialize the robot with a different rese joint position.
            obs_modalities (str or list of str): Observation modalities to use for this robot. Default is "all", which
                corresponds to all modalities being used.
                Otherwise, valid options should be part of omnigibson.sensors.ALL_SENSOR_MODALITIES.
                Note: If @sensor_config explicitly specifies `modalities` for a given sensor class, it will
                    override any values specified from @obs_modalities!
            proprio_obs (str or list of str): proprioception observation key(s) to use for generating proprioceptive
                observations. If str, should be exactly "default" -- this results in the default proprioception
                observations being used, as defined by self.default_proprio_obs. See self._get_proprioception_dict
                for valid key choices
            sensor_config (None or dict): nested dictionary mapping sensor class name(s) to specific sensor
                configurations for this object. This will override any default values specified by this class.
            kwargs (dict): Additional keyword arguments that are used for other super() calls from subclasses, allowing
                for flexible compositions of various object subclasses (e.g.: Robot is USDObject + ControllableObject).
        """
        # Store inputs
        self._obs_modalities = obs_modalities if obs_modalities == "all" else \
            {obs_modalities} if isinstance(obs_modalities, str) else set(obs_modalities)              # this will get updated later when we fill in our sensors
        self._proprio_obs = self.default_proprio_obs if proprio_obs == "default" else list(proprio_obs)
        self._sensor_config = sensor_config

        # Process abilities
        robot_abilities = {"robot": {}}
        abilities = robot_abilities if abilities is None else robot_abilities.update(abilities)

        # Initialize internal attributes that will be loaded later
        self._sensors = None                    # e.g.: scan sensor, vision sensor
        self._dummy = None                      # Dummy version of the robot w/ fixed base for computing generalized gravity forces

        # If specified, make sure scale is uniform -- this is because non-uniform scale can result in non-matching
        # collision representations for parts of the robot that were optimized (e.g.: bounding sphere for wheels)
        assert scale is None or isinstance(scale, int) or isinstance(scale, float) or np.all(scale == scale[0]), \
            f"Robot scale must be uniform! Got: {scale}"

        # Run super init
        super().__init__(
            prim_path=prim_path,
            usd_path=self.usd_path,
            name=name,
            category=m.ROBOT_CATEGORY,
            uuid=uuid,
            scale=scale,
            visible=visible,
            fixed_base=fixed_base,
            visual_only=visual_only,
            self_collisions=self_collisions,
            prim_type=PrimType.RIGID,
            include_default_states=True,
            load_config=load_config,
            abilities=abilities,
            control_freq=control_freq,
            controller_config=controller_config,
            action_type=action_type,
            action_normalize=action_normalize,
            reset_joint_pos=reset_joint_pos,
            **kwargs,
        )

    def _load(self):
        # Run super first
        prim = super()._load()

        # Also import dummy object if this robot is not fixed base
        if self._use_dummy:
            dummy_path = f"{self._prim_path}_dummy"
            dummy_prim = add_asset_to_stage(asset_path=self._dummy_usd_path, prim_path=dummy_path)
            self._dummy = BaseObject(
                name=f"{self.name}_dummy",
                prim_path=dummy_path,
                scale=self._load_config.get("scale", None),
                visible=False,
                fixed_base=True,
                visual_only=True,
            )

        return prim

    def _post_load(self):
        # Run super post load first
        super()._post_load()

        # Load the sensors
        self._load_sensors()

    def _initialize(self):
        # Initialize the dummy first if it exists
        if self._dummy is not None:
            self._dummy.initialize()

        # Run super
        super()._initialize()

        # Initialize all sensors
        for sensor in self._sensors.values():
            sensor.initialize()

        # Load the observation space for this robot
        self.load_observation_space()

        # Validate this robot configuration
        self._validate_configuration()

        self._reset_joint_pos_aabb_extent = self.aabb_extent

    def _load_sensors(self):
        """
        Loads sensor(s) to retrieve observations from this object.
        Stores created sensors as dictionary mapping sensor names to specific sensor
        instances used by this object.
        """
        # Populate sensor config
        self._sensor_config = self._generate_sensor_config(custom_config=self._sensor_config)

        # Search for any sensors this robot might have attached to any of its links
        self._sensors = dict()
        obs_modalities = set()
        for link_name, link in self._links.items():
            # Search through all children prims and see if we find any sensor
            sensor_counts = {p: 0 for p in SENSOR_PRIMS_TO_SENSOR_CLS.keys()}
            for prim in link.prim.GetChildren():
                prim_type = prim.GetPrimTypeInfo().GetTypeName()
                if prim_type in SENSOR_PRIMS_TO_SENSOR_CLS:
                    # Infer what obs modalities to use for this sensor
                    sensor_cls = SENSOR_PRIMS_TO_SENSOR_CLS[prim_type]
                    sensor_kwargs = self._sensor_config[sensor_cls.__name__]
                    if "modalities" not in sensor_kwargs:
                        sensor_kwargs["modalities"] = sensor_cls.all_modalities if self._obs_modalities == "all" else \
                            sensor_cls.all_modalities.intersection(self._obs_modalities)
                    obs_modalities = obs_modalities.union(sensor_kwargs["modalities"])
                    # Create the sensor and store it internally
                    sensor = create_sensor(
                        sensor_type=prim_type,
                        prim_path=str(prim.GetPrimPath()),
                        name=f"{self.name}:{link_name}:{prim_type}:{sensor_counts[prim_type]}",
                        **sensor_kwargs,
                    )
                    self._sensors[sensor.name] = sensor
                    sensor_counts[prim_type] += 1

        # Since proprioception isn't an actual sensor, we need to possibly manually add it here as well
        if self._obs_modalities == "all" or "proprio" in self._obs_modalities:
            obs_modalities.add("proprio")

        # Update our overall obs modalities
        self._obs_modalities = obs_modalities

    def _generate_sensor_config(self, custom_config=None):
        """
        Generates a fully-populated sensor config, overriding any default values with the corresponding values
        specified in @custom_config

        Args:
            custom_config (None or Dict[str, ...]): nested dictionary mapping sensor class name(s) to specific custom
                sensor configurations for this object. This will override any default values specified by this class

        Returns:
            dict: Fully-populated nested dictionary mapping sensor class name(s) to specific sensor configurations
                for this object
        """
        sensor_config = {} if custom_config is None else deepcopy(custom_config)

        # Merge the sensor dictionaries
        sensor_config = merge_nested_dicts(
            base_dict=self._default_sensor_config,
            extra_dict=sensor_config,
        )

        return sensor_config

    def _validate_configuration(self):
        """
        Run any needed sanity checks to make sure this robot was created correctly.
        """
        pass

    def step(self):
        # Skip this step if our articulation view is not valid
        if self._articulation_view_direct is None or not self._articulation_view_direct.initialized:
            return

        # Before calling super, update the dummy robot's kinematic state based on this robot's kinematic state
        # This is done prior to any state getter calls, since setting kinematic state results in physx backend
        # having to re-fetch tensorized state.
        # We do this so we have more optimal runtime performance
        if self._use_dummy:
            self._dummy.set_joint_positions(self.get_joint_positions())
            self._dummy.set_joint_velocities(self.get_joint_velocities())
            self._dummy.set_position_orientation(*self.get_position_orientation())

        super().step()

    def get_obs(self):
        """
        Grabs all observations from the robot. This is keyword-mapped based on each observation modality
            (e.g.: proprio, rgb, etc.)

        Returns:
            2-tuple:
                dict: Keyword-mapped dictionary mapping observation modality names to
                    observations (usually np arrays)
                dict: Keyword-mapped dictionary mapping observation modality names to
                    additional info
        """
        # Our sensors already know what observation modalities it has, so we simply iterate over all of them
        # and grab their observations, processing them into a flat dict
        obs_dict = dict()
        info_dict = dict()
        for sensor_name, sensor in self._sensors.items():
            obs_dict[sensor_name], info_dict[sensor_name] = sensor.get_obs()

        # Have to handle proprio separately since it's not an actual sensor
        if "proprio" in self._obs_modalities:
            obs_dict["proprio"], info_dict["proprio"] = self.get_proprioception()

        return obs_dict, info_dict

    def get_proprioception(self):
        """
        Returns:
            n-array: numpy array of all robot-specific proprioceptive observations.
            dict: empty dictionary, a placeholder for additional info
        """
        proprio_dict = self._get_proprioception_dict()
        return np.concatenate([proprio_dict[obs] for obs in self._proprio_obs]), {}

    def _get_proprioception_dict(self):
        """
        Returns:
            dict: keyword-mapped proprioception observations available for this robot.
                Can be extended by subclasses
        """
        joint_positions = self.get_joint_positions(normalized=False)
        joint_velocities = self.get_joint_velocities(normalized=False)
        joint_efforts = self.get_joint_efforts(normalized=False)
        pos, ori = self.get_position(), self.get_rpy()
        ori_2d = self.get_2d_orientation()
        return dict(
            joint_qpos=joint_positions,
            joint_qpos_sin=np.sin(joint_positions),
            joint_qpos_cos=np.cos(joint_positions),
            joint_qvel=joint_velocities,
            joint_qeffort=joint_efforts,
            robot_pos=pos,
            robot_ori_cos=np.cos(ori),
            robot_ori_sin=np.sin(ori),
            robot_2d_ori=ori_2d,
            robot_2d_ori_cos=np.cos(ori_2d),
            robot_2d_ori_sin=np.sin(ori_2d),
            robot_lin_vel=self.get_linear_velocity(),
            robot_ang_vel=self.get_angular_velocity(),
        )

    def _load_observation_space(self):
        # We compile observation spaces from our sensors
        obs_space = dict()

        for sensor_name, sensor in self._sensors.items():
            # Load the sensor observation space
            obs_space[sensor_name] = sensor.load_observation_space()

        # Have to handle proprio separately since it's not an actual sensor
        if "proprio" in self._obs_modalities:
            obs_space["proprio"] = self._build_obs_box_space(shape=(self.proprioception_dim,), low=-np.inf, high=np.inf, dtype=np.float64)

        return obs_space

    def add_obs_modality(self, modality):
        """
        Adds observation modality @modality to this robot. Note: Should be one of omnigibson.sensors.ALL_SENSOR_MODALITIES

        Args:
            modality (str): Observation modality to add to this robot
        """
        # Iterate over all sensors we own, and if the requested modality is a part of its possible valid modalities,
        # then we add it
        for sensor in self._sensors.values():
            if modality in sensor.all_modalities:
                sensor.add_modality(modality=modality)

    def remove_obs_modality(self, modality):
        """
        Remove observation modality @modality from this robot. Note: Should be one of
        omnigibson.sensors.ALL_SENSOR_MODALITIES

        Args:
            modality (str): Observation modality to remove from this robot
        """
        # Iterate over all sensors we own, and if the requested modality is a part of its possible valid modalities,
        # then we remove it
        for sensor in self._sensors.values():
            if modality in sensor.all_modalities:
                sensor.remove_modality(modality=modality)
    
    def get_robot_mask(self):
        """
        Returns:
            np.ndarray: 2D binary mask of the robot in the scene
        """
        mask = None
        for sensor in self.sensors.values():
            obs, _ = sensor.get_obs()
            if isinstance(sensor, VisionSensor):
                # We check for rgb, depth, normal, seg_instance
                for modality in ["seg_instance"]:
                    if modality in sensor.modalities:
                        raw_ob = obs[modality]
                        # the mask is where raw ob is 2.
                        mask = raw_ob == 2
                        return mask
    
    def get_vision_data(self):
        """
        Renders this robot's key sensors, visualizing them via matplotlib plots
        """
        remaining_obs_modalities = deepcopy(self.obs_modalities)
        
        if len(self.sensors) > 1:
            raise ValueError("This method only supports robots with a single sensor for now!")
        
        for sensor in self.sensors.values():
            obs, _ = sensor.get_obs()
            sensor_frames = {}
            if isinstance(sensor, VisionSensor):
                # We check for rgb, depth, normal, seg_instance
                for modality in ["rgb", "depth", "normal", "seg_instance", "seg_semantic"]:
                    if modality in sensor.modalities:
                        ob = obs[modality]
                        if modality == "rgb":
                            # Ignore alpha channel, map to floats
                            ob = ob[:, :, :3] / 255.0
                        elif modality == "seg_instance" or modality == "seg_semantic":
                            # Map IDs to rgb
                            ob = segmentation_to_rgb(ob, N=256) / 255.0
                        elif modality == "normal":
                            # Re-map to 0 - 1 range
                            ob = (ob + 1.0) / 2.0
                        else:
                            # Depth, nothing to do here
                            pass
                        # Add this observation to our frames and remove the modality
                        sensor_frames[modality] = ob
                        remaining_obs_modalities -= {modality}
                    else:
                        # Warn user that we didn't find this modality
                        print(f"Modality {modality} is not active in sensor {sensor.name}, skipping...")

            return sensor_frames

    def visualize_sensors(self):
        """
        Renders this robot's key sensors, visualizing them via matplotlib plots
        """
        frames = dict()
        remaining_obs_modalities = deepcopy(self.obs_modalities)
        for sensor in self.sensors.values():
            obs, _ = sensor.get_obs()
            sensor_frames = []
            if isinstance(sensor, VisionSensor):
                # We check for rgb, depth, normal, seg_instance
                for modality in ["rgb", "depth", "normal", "seg_instance", "seg_semantic"]:
                    if modality in sensor.modalities:
                        ob = obs[modality]
                        if modality == "rgb":
                            # Ignore alpha channel, map to floats
                            ob = ob[:, :, :3] / 255.0
                        elif modality == "seg_instance" or modality == "seg_semantic":
                            # Map IDs to rgb
                            ob = segmentation_to_rgb(ob, N=256) / 255.0
                        elif modality == "normal":
                            # Re-map to 0 - 1 range
                            ob = (ob + 1.0) / 2.0
                        else:
                            # Depth, nothing to do here
                            pass
                        # Add this observation to our frames and remove the modality
                        sensor_frames.append((modality, ob))
                        remaining_obs_modalities -= {modality}
                    else:
                        # Warn user that we didn't find this modality
                        print(f"Modality {modality} is not active in sensor {sensor.name}, skipping...")
            elif isinstance(sensor, ScanSensor):
                # We check for occupancy_grid
                occupancy_grid = obs.get("occupancy_grid", None)
                if occupancy_grid is not None:
                    sensor_frames.append(("occupancy_grid", occupancy_grid))
                    remaining_obs_modalities -= {"occupancy_grid"}

            # Map the sensor name to the frames for that sensor
            frames[sensor.name] = sensor_frames

        # Warn user that any remaining modalities are not able to be visualized
        if len(remaining_obs_modalities) > 0:
            print(f"Modalities: {remaining_obs_modalities} cannot be visualized, skipping...")

        # Write all the frames to a plot
        for sensor_name, sensor_frames in frames.items():
            n_sensor_frames = len(sensor_frames)
            if n_sensor_frames > 0:
                fig, axes = plt.subplots(nrows=1, ncols=n_sensor_frames)
                if n_sensor_frames == 1:
                    axes = [axes]
                # Dump frames and set each subtitle
                for i, (modality, frame) in enumerate(sensor_frames):
                    axes[i].imshow(frame)
                    axes[i].set_title(modality)
                    axes[i].set_axis_off()
                # Set title
                fig.suptitle(sensor_name)
                plt.show(block=False)

        # One final plot show so all the figures get rendered
        plt.show()

    def update_handles(self):
        # Call super first
        super().update_handles()

        # If we have a dummy robot, also update its handles too
        if self._dummy is not None:
            self._dummy.update_handles()

    def remove(self):
        """
        Do NOT call this function directly to remove a prim - call og.sim.remove_prim(prim) for proper cleanup
        """
        # Remove all sensors
        for sensor in self._sensors.values():
            sensor.remove()

        # Run super
        super().remove()
    
    @property
    def reset_joint_pos_aabb_extent(self):
        """
        This is the aabb extent of the robot in the robot frame after resetting the joints.
        Returns:
            3-array: Axis-aligned bounding box extent of the robot base
        """
        return self._reset_joint_pos_aabb_extent

    def teleop_data_to_action(self, teleop_action) -> np.ndarray:
        """
        Generate action data from teleoperation action data
        Args:
            teleop_action (TeleopAction): teleoperation action data
        Returns:
            np.ndarray: array of action data filled with update value
        """
        return np.zeros(self.action_dim)

    def get_generalized_gravity_forces(self, clone=True):
        # Override method based on whether we're using a dummy or not
        return self._dummy.get_generalized_gravity_forces(clone=clone) \
            if self._use_dummy else super().get_generalized_gravity_forces(clone=clone)

    @property
    def sensors(self):
        """
        Returns:
            dict: Keyword-mapped dictionary mapping sensor names to BaseSensor instances owned by this robot
        """
        return self._sensors

    @property
    def obs_modalities(self):
        """
        Returns:
            set of str: Observation modalities used for this robot (e.g.: proprio, rgb, etc.)
        """
        assert self._loaded, "Cannot check observation modalities until we load this robot!"
        return self._obs_modalities

    @property
    def proprioception_dim(self):
        """
        Returns:
            int: Size of self.get_proprioception() vector
        """
        return len(self.get_proprioception()[0])

    @property
    def _default_sensor_config(self):
        """
        Returns:
            dict: default nested dictionary mapping sensor class name(s) to specific sensor
                configurations for this object. See kwargs from omnigibson/sensors/__init__/create_sensor for more
                details

                Expected structure is as follows:
                    SensorClassName1:
                        modalities: ...
                        enabled: ...
                        noise_type: ...
                        noise_kwargs:
                            ...
                        sensor_kwargs:
                            ...
                    SensorClassName2:
                        modalities: ...
                        enabled: ...
                        noise_type: ...
                        noise_kwargs:
                            ...
                        sensor_kwargs:
                            ...
                    ...
        """
        return {
            "VisionSensor": {
                "enabled": True,
                "noise_type": None,
                "noise_kwargs": None,
                "sensor_kwargs": {
                    "image_height": 720,
                    "image_width": 720,
                },
            },
            "ScanSensor": {
                "enabled": True,
                "noise_type": None,
                "noise_kwargs": None,
                "sensor_kwargs": {

                    # Basic LIDAR kwargs
                    "min_range": 0.05,
                    "max_range": 10.0,
                    "horizontal_fov": 360.0,
                    "vertical_fov": 1.0,
                    "yaw_offset": 0.0,
                    "horizontal_resolution": 1.0,
                    "vertical_resolution": 1.0,
                    "rotation_rate": 0.0,
                    "draw_points": False,
                    "draw_lines": False,

                    # Occupancy Grid kwargs
                    "occupancy_grid_resolution": 128,
                    "occupancy_grid_range": 5.0,
                    "occupancy_grid_inner_radius": 0.5,
                    "occupancy_grid_local_link": None,
                },
            },
        }

    @property
    def default_proprio_obs(self):
        """
        Returns:
            list of str: Default proprioception observations to use
        """
        return []

    @property
    def model_name(self):
        """
        Returns:
            str: name of this robot model. usually corresponds to the class name of a given robot model
        """
        return self.__class__.__name__

    @property
    @abstractmethod
    def usd_path(self):
        # For all robots, this must be specified a priori, before we actually initialize the USDObject constructor!
        # So we override the parent implementation, and make this an abstract method
        raise NotImplementedError

    @property
    def _dummy_usd_path(self):
        """
        Returns:
            str: Absolute path to the dummy USD to load for, e.g., computing gravity compensation
        """
        # By default, this is just the normal usd path
        return self.usd_path

    @property
    def urdf_path(self):
        """
        Returns:
            str: file path to the robot urdf file.
        """
        raise NotImplementedError

    @property
    def _use_dummy(self):
        """
        Returns:
            bool: Whether the robot dummy should be loaded and used for some computations, e.g., gravity compensation
        """
        # By default, only load if robot is not fixed base
        return not self.fixed_base

    @classproperty
    def _do_not_register_classes(cls):
        # Don't register this class since it's an abstract template
        classes = super()._do_not_register_classes
        classes.add("BaseRobot")
        return classes

    @classproperty
    def _cls_registry(cls):
        # Global robot registry -- override super registry
        global REGISTERED_ROBOTS
        return REGISTERED_ROBOTS
