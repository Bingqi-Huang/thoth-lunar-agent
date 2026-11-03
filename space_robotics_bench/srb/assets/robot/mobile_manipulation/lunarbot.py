import math
from typing import ClassVar, Sequence

from srb.core.action import (
    ActionGroup,
    BinaryJointPositionActionCfg,
    DifferentialIKControllerCfg,
    FourWheelSteerActionCfg,
    SwitchableArmActionCfg,
    WheeledManipulatorActionGroup,
)
from srb.core.actuator import ImplicitActuatorCfg
from srb.core.asset import ArticulationCfg, Frame, Transform, WheeledManipulator
from srb.core.domain import Domain
from srb.core.sim import (
    ArticulationRootPropertiesCfg,
    UsdFileCfg,
)
from srb.utils.math import deg_to_rad, rpy_to_quat
from srb.utils.path import SRB_ASSETS_DIR_SRB_ROBOT

## The arm's drive gains are handed to PhysX per radian and applied per degree: a bare-simulation test of this arm (elbow
## kicked 0.05 rad, nothing else in the scene, base fixed) rang at 1.0 Hz with a damping ratio of 0.05 -- a stiffness of
## 600 * pi/180 = 10.5 N m/rad and a damping of 7.65 * pi/180 = 0.13 N m s/rad against joint inertias of 0.01-0.5 kg m^2 -- and
## settled in 0.3 s with no ringing once both were multiplied by 180/pi (research repo F98). So the values below are written
## in N m/rad and N m s/rad and converted here, once, for the arm.
_DRIVE_GAIN_PER_RAD = 180.0 / math.pi

## Tool-centre-point offset from the Link7 flange along its own +Z axis
## (the EG2-4C2 gripper is welded directly to Link7 with an identity transform)
_TCP_OFFSET = (0.0, 0.0, 0.140)

## EG2-4C2 gripper: single actuated joint (eg2_joint1) + 5 joints driven in
## lock-step (mimic behavior emulated at the action-group level, since the
## PhysX mimic constraint was dropped from the imported gripper for
## simulation stability -- see assets/srb_assets robot/mobile_manipulation/lunarbot)
_GRIPPER_STROKE_RAD = 0.82
## pad-to-pad contact of the empty jaws (measured 0.114 rad, F50); the closed command target and
## openral's `GRIPPER_CLOSED_RAD` (openral_rskill._lunar_bot_arm) must agree
_GRIPPER_CLOSED_RAD = 0.10
_GRIPPER_JOINT_NAMES = [
    "eg2_joint1",
    "eg2_joint2",
    "eg2_joint3",
    "eg2_joint4",
    "eg2_joint5",
    "eg2_joint6",
]
_GRIPPER_SIGN = {
    "eg2_joint1": 1.0,
    "eg2_joint2": 1.0,
    "eg2_joint5": 1.0,
    "eg2_joint6": 1.0,
    "eg2_joint3": -1.0,
    "eg2_joint4": -1.0,
}


class LunarBot(WheeledManipulator):
    """
    LunarBot: a four-wheel independent-drive/independent-steering (4WIS/4WID)
    rover chassis with a 7-DoF RealMan RM-75 manipulator and an Inspire-Robots
    EG2-4C2 parallel electric gripper, fused into a **single PhysX
    articulation** (base + arm + gripper share one USD/articulation root).
    """

    ## Scenario
    DOMAINS: ClassVar[Sequence[Domain]] = (Domain.MOON,)

    ## Model
    asset_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/lunarbot",
        spawn=UsdFileCfg(
            usd_path=SRB_ASSETS_DIR_SRB_ROBOT.joinpath("mobile_manipulation")
            .joinpath("lunarbot")
            .joinpath("lunarbot.usdc")
            .as_posix(),
            activate_contact_sensors=True,
            ## NOTE: gravity is disabled on the arm/gripper links directly in
            ## the USD asset (matching Franka's own convention in SRB). This
            ## is required, not cosmetic: the arm's relative-mode differential
            ## IK action re-syncs its target to the *current* joint state
            ## every step, so it provides zero position-holding stiffness
            ## against persistent external disturbances like gravity -- a
            ## real RM-75 controller performs active gravity compensation
            ## internally, which this approximates.
            articulation_props=ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.05),
            joint_pos={
                ## Steering neutral, wheels free
                "chassis_to_.*_steering_joint": 0.0,
                ## Arm at rest: reaching back over the rear hull edge with the wrist camera level, which is
                ## where this arm works from and what it should be doing when nothing is commanding it. The old
                ## spawn was the travel pose, which folds the arm up over the deck with the camera pointing
                ## forward and 21 deg above the horizon -- and until a skill first claimed the arm the action
                ## buffer's zeros selected twist mode against a desired pose that had never been set, so the arm
                ## chased it at its velocity limit for the first 150 s of every run (F89). Measured from this
                ## robot's own URDF: tool 0.694 m behind the rover origin at 1.404 m -- 26 mm past the rear hull
                ## face, so it does not change what the rover can back up to -- look 0.0 deg from level
                ## and 180 deg from forward, elbow bent 40 deg, and on the working branch the skills
                ## seed their IK from (the old pose was off it). Wrist rolled to pi/2 so that the wrist camera's
                ## image stands upright, not on its side (research repo F141); the tool's position and axis are
                ## unchanged, the fingers open sideways.
                "joint1": 0.0,
                "joint2": 0.40,
                "joint3": 0.0,
                "joint4": -0.70,
                "joint5": 0.0,
                "joint6": -1.27,
                "joint7": 1.57079633,
                ## Gripper: open
                "eg2_joint1": _GRIPPER_STROKE_RAD,
                "eg2_joint2": _GRIPPER_STROKE_RAD,
                "eg2_joint5": _GRIPPER_STROKE_RAD,
                "eg2_joint6": _GRIPPER_STROKE_RAD,
                "eg2_joint3": -_GRIPPER_STROKE_RAD,
                "eg2_joint4": -_GRIPPER_STROKE_RAD,
            },
        ),
        actuators={
            "steering": ImplicitActuatorCfg(
                joint_names_expr=["chassis_to_.*_steering_joint"],
                effort_limit=100.0,
                velocity_limit=1.9635,
                stiffness=280.0,
                damping=90.0,
            ),
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=[".*_steering_to_wheel_joint"],
                effort_limit=48.512,
                velocity_limit=41.4458,
                stiffness=0.0,
                damping=40.0,
            ),
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-7]"],
                effort_limit={
                    "joint1": 60.0,
                    "joint2": 60.0,
                    "joint3": 30.0,
                    "joint4": 30.0,
                    "joint5": 10.0,
                    "joint6": 10.0,
                    "joint7": 10.0,
                },
                velocity_limit={
                    "joint1": 3.14,
                    "joint2": 3.14,
                    "joint3": 3.92,
                    "joint4": 3.92,
                    "joint5": 3.92,
                    "joint6": 3.92,
                    "joint7": 3.92,
                },
                stiffness=600.0 * _DRIVE_GAIN_PER_RAD,
                ## Damping per joint = effort limit / velocity limit: a joint at its rated speed must not spend more
                ## than its rated torque on damping alone. The former 140 N m s/rad capped the wrist (10 N m) at
                ## 0.07 rad/s, so every arm motion lagged 0.3-1.4 s at the torque limit (research repo F78).
                damping={
                    "joint1": 60.0 / 3.14 * _DRIVE_GAIN_PER_RAD,
                    "joint2": 60.0 / 3.14 * _DRIVE_GAIN_PER_RAD,
                    "joint3": 30.0 / 3.92 * _DRIVE_GAIN_PER_RAD,
                    "joint4": 30.0 / 3.92 * _DRIVE_GAIN_PER_RAD,
                    "joint5": 10.0 / 3.92 * _DRIVE_GAIN_PER_RAD,
                    "joint6": 10.0 / 3.92 * _DRIVE_GAIN_PER_RAD,
                    "joint7": 10.0 / 3.92 * _DRIVE_GAIN_PER_RAD,
                },
            ),
            # Every jaw joint is driven AND the PhysX mimic joints in lunarbot.usdc lock them to
            # eg2_joint1 (ratios +1/-1, the signs that keep the pads parallel). The constraint is what
            # makes both jaws clamp together the way the real EG2's single actuator + linkage does;
            # the drives supply the force. Drives alone let one side close fully while the other was
            # levered open under load, and a single driven joint could not hold the coupled mechanism
            # against a 1 kg payload (research repo F49).
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=_GRIPPER_JOINT_NAMES,
                # ~65 N at the pads (0.06 m lever), inside the EG2-4C2's grip-force band. At the
                # previous 0.9 N.m (~15 N) a 1 kg payload hanging 0.2 m below the jaws levered the
                # finger back open mid-lift while the drive sat saturated (research repo F49).
                effort_limit=4.0,
                velocity_limit=_GRIPPER_STROKE_RAD / 1.3,
                stiffness=400.0,
                damping=20.0,
                armature=1.0e-4,
            ),
        },
    )

    ## Actions
    actions: ActionGroup = WheeledManipulatorActionGroup(
        cmd_vel=FourWheelSteerActionCfg(
            asset_name="robot",
            steering_joint_names=[
                "chassis_to_front_left_steering_joint",
                "chassis_to_front_right_steering_joint",
                "chassis_to_rear_left_steering_joint",
                "chassis_to_rear_right_steering_joint",
            ],
            drive_joint_names=[
                "front_left_steering_to_wheel_joint",
                "front_right_steering_to_wheel_joint",
                "rear_left_steering_to_wheel_joint",
                "rear_right_steering_to_wheel_joint",
            ],
            ## (x, y) of each wheel in the chassis body frame [m], same order as above
            ## NOTE: the rear pair is offset an extra -0.03 m in x relative to
            ## the URDF's (symmetric) value -- the rear axle's USD rest pose
            ## was shifted rearward by that amount to correct a mismatch
            ## against the (asymmetric) chassis hull/rail mesh; keep this in
            ## sync with the rear steering/wheel links' authored positions.
            wheel_positions=[
                (0.4925, 0.42705),
                (0.4925, -0.42705),
                (-0.5225, 0.42705),
                (-0.5225, -0.42705),
            ],
            wheel_radius=0.1453,
            max_steering_angle=deg_to_rad(120.0),
            ## (v, omega) in chassis_base_link, REP-103: +v drives toward +X (the
            ## front mast with the RGB-D camera and Mid-360), +omega yaws
            ## counter-clockwise about +Z. Both scales were -1.0 until F43:
            ## +v then drove toward the arm end (-X) while every consumer
            ## (Nav2, odom TF, sensor frames) treats +X as forward, and +omega
            ## yawed about -Z -- a yaw-rate sign cannot depend on which end is
            ## called forward, so that negation was wrong under any convention.
            ## Measured on lunar_base: omega=+0.5 gave -0.305 rad/s world z.
            scale_linear=1.0,
            scale_angular=1.0,
        ),
        arm=SwitchableArmActionCfg(
            asset_name="robot",
            joint_names=[
                "joint1",
                "joint2",
                "joint3",
                "joint4",
                "joint5",
                "joint6",
                "joint7",
            ],
            # IK twist commands are expressed in the articulation root (chassis_base_link): that
            # is the frame OpenRAL's CARTESIAN_TWIST contract names (rskill-procedural-move_ee_to_pose
            # `reference_frame`, `openral_hal.lunar_bot_srb`), and the rover's TF tree is rooted
            # there. The arm's own `base_link` body is mounted at the rear, so a servo whose error
            # is measured in the chassis frame would otherwise be commanding in a rotated frame.
            base_name="",
            body_name="Link7",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="dls",
            ),
            # per control step (0.02 s) displacement per raw unit, so a raw unit is 0.5 m/s and
            # 0.6 rad/s -- the calibration openral_hal.lunar_bot_srb converts physical twists with.
            # The twist is integrated onto a persistent desired pose (switchable_arm.py); the old
            # 0.1 scale on a re-synced relative delta realised only 1.5-4 % of the command (F50).
            ik_scale=0.01,
            ik_scale_angular=0.012,
            body_offset=SwitchableArmActionCfg.OffsetCfg(pos=_TCP_OFFSET),
            # Direct radians pass-through: raw joint-position sub-command IS
            # the physical target angle, matching OpenRAL's JOINT_POSITION
            # control mode's own convention (physical units end to end, no
            # HAL-side unit conversion needed -- unlike the IK sub-mode's
            # raw<->m/s calibration, which SRB's differential-IK action term
            # genuinely needs since it treats its raw input as a scaled
            # per-step position delta, not a physical unit).
            joint_pos_scale=1.0,
        ),
        hand=BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=_GRIPPER_JOINT_NAMES,
            # closed = the pads' contact position, not 0: commanded past contact the two pads are
            # driven into each other and chatter (empty jaws never settled, research repo F50)
            close_command_expr={jn: sign * _GRIPPER_CLOSED_RAD for jn, sign in _GRIPPER_SIGN.items()},
            open_command_expr={
                jn: sign * _GRIPPER_STROKE_RAD for jn, sign in _GRIPPER_SIGN.items()
            },
        ),
    )

    ## Frames
    frame_base: Frame = Frame(prim_relpath="chassis_base_link")
    ## Livox Mid-360 lidar mount, front mast (see docs/research_findings.md F23
    ## for the CAD calibration/provenance of this pose -- Y/Z are cross-checked
    ## against independent geometric evidence (chassis symmetry, wheel radius),
    ## X is anchored to the existing camera mast (medium confidence, pending a
    ## live Isaac Sim visual check). Orientation is the lidar mounted inverted
    ## (dome down through the mount box's floor hole, base flange up into the
    ## box cavity), derived from the CAD, not assumed. The prim's own axes are
    ## the sensor axes (+Z = scan axis, pointing world-down, roll 180 deg), so the
    ## RayCaster's -7..+52 deg pattern lands at -52..+7 deg in the world and the
    ## unscanned cone below -52 deg encloses the vehicle (F42); the child mesh
    ## carries its own Y-up correction.
    frame_lidar: Frame = Frame(prim_relpath="lidar_mid360_frame")
    frame_flange: Frame = Frame(prim_relpath="Link7", offset=Transform(pos=_TCP_OFFSET))
    frame_front_camera: Frame = Frame(
        prim_relpath="rgbd_camera_frame/camera_front",
        offset=Transform(rot=rpy_to_quat(0.0, 15.0, 0.0)),
    )
    ## Wrist RGB-D (D435i) on a bracket beside the gripper: 0.12 m out from the tool
    ## axis, 0.22 m behind the TCP (0.08 m behind the flange), its optical axis tilted
    ## 18 deg towards that axis so it crosses it 0.15 m in front of the grasp point.
    ## Computed, not tuned (research repo F57): with the D435i's own 640x480 colour
    ## intrinsics this leaves the gripper in the bottom 40% of the image with both pad
    ## tips visible, and puts the part being approached in the middle of the frame. The
    ## earlier mount (0.048 m out, aimed at the grasp point) filled two thirds of every
    ## frame with the fingers.
    frame_wrist_camera: Frame = Frame(
        prim_relpath="Link7/camera_wrist",
        offset=Transform(
            pos=(0.0, -0.12, -0.08),
            rot=rpy_to_quat(0.0, -72.0, 90.0),
        ),
    )
