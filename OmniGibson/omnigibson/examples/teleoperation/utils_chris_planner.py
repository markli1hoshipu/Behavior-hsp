from b1k.planner.whole_body import WholeBodyPlanner
from b1k.ik import DualArmIKController, GlobalDualArmIK
from b1k.models.r1pro.constants import r1pro_init_joint_state
from b1k.planner.full_planner import FullBodyPlanner
from typing import Dict, List, Optional, Tuple
import torch as th
import numpy as np
from math import radians, degrees

from b1k.planner.nav import NavigationPlanner

########### RH Magic
def corrected_command_z(desired_rz_deg):
    sign = np.sign(desired_rz_deg)
    desired_rz_deg = np.abs(desired_rz_deg)
    a = 3.1008740730597877
    b = -3.1049333609562475
    return sign * (desired_rz_deg - b) / a

def corrected_command(desired_x):
    sign = np.sign(desired_x)
    desired_x = np.abs(desired_x)
    a2, a1, a0 = 0.999, 1.187, -0.015  # from your fit
    return sign * (-a1 + np.sqrt(a1**2 - 4*a2*(a0 - desired_x))) / (2*a2)
########### RH Magic


class ChrisIKAdapter:
    """Adapter for Chris's IK solver. Wraps DualArmIKController"""

    def __init__(self):
        self.dt = 1.0 / 50.0
        self.ik = DualArmIKController(self.dt)
        self.qn = [r1pro_init_joint_state[n] for n in self.ik.joint_names]

        # Planner setup
        self.T_plan = 30
        self.planner = WholeBodyPlanner(self.T_plan)
        self.global_ik = GlobalDualArmIK()

        self.nav_plan = NavigationPlanner(30)

        from b1k.ik import GlobalSingleArmIK
        self.global_single_ik_left = GlobalSingleArmIK('left')
        self.global_single_ik_right = GlobalSingleArmIK('right')

        self.full_planner = FullBodyPlanner()


        print("[ChrisIKAdapter] Initialized")

    def forward_barge_through_door(self, arm: str, qc: dict, qc_: list):
        
        if arm == 'left':
            ik = self.global_single_ik_left
        elif arm == 'right':
            ik = self.global_single_ik_right
        else:
            raise ValueError("Invalid arm specified. Must be 'left' or 'right'.")        

        T = ik.chain.forward_kinematics(qc)
        rG = T[f'{arm}_ee_link'].rotation().as_quat().toarray().flatten().tolist()

        

        pG = [1, 0, 1]
        left_goal = (th.tensor(pG), th.tensor(rG))
        right_goal = (th.tensor(pG), th.tensor(rG))

        actions_out, _ = self.solve_dual_arm_trajectory(            
            qc_,
            left_goal,
            right_goal,
            arms = arm,
        )

        #######################################################################################################
        ## Keep for now
        #######################################################################################################
        # if arm == 'left':
        #     ik = self.global_single_ik_left
        # elif arm == 'right':
        #     ik = self.global_single_ik_right
        # else:
        #     raise ValueError("Invalid arm specified. Must be 'left' or 'right'.")
        
        # config = {
        #     "pG": [1.0, 0.0, 1.0],
        #     "rG": rG,
        #     "q0": [qc[n] for n in ik.joint_names],
        #     "p_err_max": 1e-8,
        #     "r_err_max": 1e-8,
        #     "maintain_gaze": 10.0,
        # }
        # ik.reset(config)
        # if ik.solve():
        #     print("Global IK solved")
        # else:
        #     raise RuntimeError("global ik failed!")
        
        # qsol = ik.get_solution()
        # qF = []
        # for n in self.planner.joint_names:
        #     if n in qsol:
        #         qF.append(qsol[n])
        #     else:
        #         qF.append(qc[n])

        # duration = 30.0
        # t = np.linspace(0, duration, self.T_plan)
        # Q0 = np.linspace([qc[n] for n in self.planner.joint_names], qF, self.T_plan)
        # dQ0 = np.gradient(Q0, t, axis=0)
        # ddQ0 = np.gradient(dQ0, t, axis=0)

        # config = {
        #     "Q0": Q0,
        #     "dQ0": dQ0,
        #     "ddQ0": ddQ0,
        #     "duration": duration,
        #     "q0": [qc[n] for n in self.planner.joint_names],
        #     "qF": qF,
        #     "w_dQ": 1.0,
        #     "w_ddQ": 50.0,
        # }

        # self.planner.reset(config)
        # if self.planner.solve():
        #     print("[PLANNER] Solved successfully")
        # else:
        #     raise RuntimeError("planner failed")
        
        # #qsol, _, _ = self.planner.get_solution()

        # controller_info = {
        #     'base': {'start_idx': 0, 'dofs': np.array([0, 1, 5]), 'command_dim': 3},
        #     'trunk': {'start_idx': 3, 'dofs': np.array([6, 7, 8, 9]), 'command_dim': 4},
        #     'arm_left': {'start_idx': 7, 'dofs': np.array([10, 12, 14, 16, 18, 20, 22]), 'command_dim': 7},
        #     'gripper_left': {'start_idx': 14, 'dofs': np.array([24, 25]), 'command_dim': 1},
        #     'arm_right': {'start_idx': 15, 'dofs': np.array([11, 13, 15, 17, 19, 21, 23]), 'command_dim': 7},
        #     'gripper_right': {'start_idx': 22, 'dofs': np.array([26, 27]), 'command_dim': 1}
        # }

        # joint_names_use = ['torso_joint1', 'torso_joint2', 'torso_joint3', 'torso_joint4']
        # for i in range(1, 8):
        #     joint_names_use.append(f"left_arm_joint{i}")
        # for i in range(1, 8):
        #     joint_names_use.append(f"right_arm_joint{i}")          

        # print("joint_names_use length is", len(joint_names_use)  )

        # joint_names = ['torso_joint1', 'torso_joint2', 'torso_joint3', 'torso_joint4', 'left_arm_joint1', 'right_arm_joint1', 'left_arm_joint2', 'right_arm_joint2', 'left_arm_joint3', 'right_arm_joint3', 'left_arm_joint4', 'right_arm_joint4', 'left_arm_joint5', 'right_arm_joint5', 'left_arm_joint6', 'right_arm_joint6', 'left_arm_joint7', 'right_arm_joint7', 'left_gripper_finger_joint1', 'left_gripper_finger_joint2', 'right_gripper_finger_joint1', 'right_gripper_finger_joint2']
        # joint_limits = {'torso_joint1': (-1.1344999074935913, 1.8325998783111572), 'torso_joint2': (-2.7924997806549072, 2.5306997299194336), 'torso_joint3': (-1.8325998783111572, 1.5707999467849731), 'torso_joint4': (-3.054299831390381, 3.054299831390381), 'left_arm_joint1': (-4.4506001472473145, 1.309000015258789), 'right_arm_joint1': (-4.4506001472473145, 1.309000015258789), 'left_arm_joint2': (-0.1745000034570694, 3.1415998935699463), 'right_arm_joint2': (-3.1415998935699463, 0.1745000034570694), 'left_arm_joint3': (-2.3561956882476807, 2.3561956882476807), 'right_arm_joint3': (-2.3561956882476807, 2.3561956882476807), 'left_arm_joint4': (-2.0943996906280518, 0.3490999639034271), 'right_arm_joint4': (-2.0943996906280518, 0.3490999639034271), 'left_arm_joint5': (-2.3561956882476807, 2.3561956882476807), 'right_arm_joint5': (-2.3561956882476807, 2.3561956882476807), 'left_arm_joint6': (-1.0471980571746826, 1.0471980571746826), 'right_arm_joint6': (-1.0471980571746826, 1.0471980571746826), 'left_arm_joint7': (-1.5707999467849731, 1.5707999467849731), 'right_arm_joint7': (-1.5707999467849731, 1.5707999467849731), 'left_gripper_finger_joint1': (0.0, 0.05000000074505806), 'left_gripper_finger_joint2': (0.0, 0.05000000074505806), 'right_gripper_finger_joint1': (0.0, 0.05000000074505806), 'right_gripper_finger_joint2': (0.0, 0.05000000074505806)}
        
        # if joint_names is not None:
        #     joint_names = joint_names
        # if joint_limits is not None:
        #     joint_limits = joint_limits        

        # debug = False
        
        # # Convert into actions
        # Qsol, _, _ = self.planner.get_solution()

        # # Sample trajectory at desired resolution
        # num_targets = 30
        # t = np.linspace(0, duration, num_targets)
        # q_targets = [
        #     {n: Qsol[n](t[i]) for n in self.planner.joint_names}
        #     for i in range(num_targets)
        # ]
        # if debug:
        #     print(q_targets[-1])
        #     print("++++++++++ SOLVED PLANNER 2 +++++++++++")

        # # Convert joint targets to action sequence
        # ctrl_groups = ["trunk", "arm_left", "arm_right"]
        # actions_out = [th.zeros(23) for _ in range(num_targets)]

        # normalize_actions: bool = False

        # for i in range(num_targets):
        #     q_targ = q_targets[i]
        #     for grp in ctrl_groups:
        #         if grp not in controller_info:
        #             continue

        #         dof_idx = controller_info[grp]["dofs"]
        #         start = controller_info[grp]["start_idx"]
        #         for local_i, j_idx in enumerate(dof_idx.tolist()):
        #             j_name = joint_names[j_idx-6]
        #             # print(j_name)
        #             if j_name in q_targ:
        #                 if normalize_actions:
        #                     # Convert absolute joint position to normalized [-1, 1] command
        #                     normalized_cmd = self.normalize_joint_position(
        #                         j_name, float(q_targ[j_name]), joint_limits
        #                     )
        #                     actions_out[i][start + local_i] = normalized_cmd
        #                 else:
        #                     # Use absolute joint position directly
        #                     actions_out[i][start + local_i] = float(q_targ[j_name])
        #######################################################################################################



        duration = 30.0
        # Now move base
        if arm == 'left':
            goal_base = [1., 0.0, 0.0]
        elif arm == 'right':
            goal_base = [1., 0.0, 0.0] # TODO update if needed            


        goal1 = [corrected_command(goal_base[0]), corrected_command(goal_base[1]), 0.0]

        config = {"duration": duration, "goal": goal1}
        self.nav_plan.reset(config)
        if self.nav_plan.solve():
            print("First planner solver")
        else:
            raise RuntimeError("nav planner failed")

        vx_lin, vy_lin, w_lin = self.nav_plan.get_solution()

        goal2 = [0, 0, radians(corrected_command_z(degrees(goal_base[2])))]
        print(goal2)
        config = {"duration": duration, "goal": goal2}
        self.nav_plan.reset(config)
        if self.nav_plan.solve():
            print("Second planner solver")
        else:
            raise RuntimeError("second planner failed")

        vx_ang, vy_ang, w_ang = self.nav_plan.get_solution()

        actions = []

        t = 0.0
        dt = 1.0 / 30.0
        while t < duration:
            act = np.asarray(actions_out[-1]) #np.zeros(23, dtype=float)
            act[0] = vx_lin(t)
            act[1] = vy_lin(t)
            act[2] = w_lin(t)
            # act[3:] = actions_out[-1]
            actions.append(act.copy())
            t += dt

        t = 0.0
        while t < duration:
            act = np.asarray(actions_out[-1]) #np.zeros(23, dtype=float)
            act[0] = vx_ang(t)
            act[1] = vy_ang(t)
            act[2] = w_ang(t)
            # act[3:] = qc
            actions.append(act.copy())
            t += dt
    

        return actions_out + actions


    def get_current_joint_state(self, joint_positions: th.Tensor, joint_names: list) -> list:
        """Convert robot joint positions to IK joint state.

        Args:
            joint_positions: Tensor of current joint positions from robot
            joint_names: List of joint names from robot

        Returns:
            List of joint positions in the order expected by the IK solver
        """
        index = {n: i for i, n in enumerate(joint_names)}
        q_dict = {}
        for n in self.ik.joint_names:
            if n in index:
                q_dict[n] = float(joint_positions[index[n]])
            else:
                q_dict[n] = 0.0
        # breakpoint()
        return [q_dict[n] for n in self.ik.joint_names]

    def solve_to_joint_targets(
        self,
        current_joint_state: list,
        target_pose_base_left: Tuple[th.Tensor, th.Tensor],
        target_pose_base_right: Tuple[th.Tensor, th.Tensor],
    ) -> Optional[Dict[str, float]]:
        """Solve IK for dual-arm targets.

        Args:
            current_joint_state: Current joint state from get_current_joint_state
            target_pose_base_left: Left arm target (position, quaternion)
            target_pose_base_right: Right arm target (position, quaternion)

        Returns:
            Dictionary mapping joint names to target positions
        """
        def parse_target(target_pose_base):
            pos_t, quat_t = target_pose_base
            pos = pos_t.detach().cpu().numpy().astype("float64")
            quat = quat_t.detach().cpu().numpy().astype("float64")
            return pos, quat

        pG_left, rG_left = parse_target(target_pose_base_left)
        pG_right, rG_right = parse_target(target_pose_base_right)

        config = {
            "pG_left": pG_left,
            "pG_right": pG_right,
            "rG_left": rG_left,
            "rG_right": rG_right,
            "w_dq": 0.01,
            "w_qn": 1e5,
            "qn": self.qn,
            "w_p": 1e8,
            "w_r": 1e8,
            "w_gaze": 1e6,
            "q": current_joint_state,
        }
        self.ik.reset(config)
        if self.ik.solve():
            dq = self.ik.get_solution()
        else:
            print("[IK] Solver failed")
            dq = np.zeros(self.ik.dof)
        qsol = current_joint_state + self.dt * dq
        return {n: qsol[i] for i, n in enumerate(self.ik.joint_names)}

    def solve_dual_arm_trajectory(
        self,
        current_joint_positions: list,
        left_goal: Tuple[th.Tensor, th.Tensor],
        right_goal: Tuple[th.Tensor, th.Tensor],
        duration: float = 30,
        num_targets: int = 30,
        arms = 'both',
        normalize_actions: bool = False,
        joint_names: list = None,
        joint_limits: dict = None,
        debug: bool = False,
    ) -> Tuple[list, list]:
        """Solve global IK and motion planning for dual-arm trajectory.

        Args:
            current_joint_positions: Current joint positions as a list (from get_current_joint_state)
            left_goal: Tuple of (position, quaternion) for left end-effector goal
            right_goal: Tuple of (position, quaternion) for right end-effector goal
            base_action: Base action tensor to clone for each trajectory point
            joint_names: List of robot joint names
            joint_limits: Dictionary mapping joint names to (lower_limit, upper_limit) tuples
            duration: Trajectory duration in seconds
            num_targets: Number of trajectory waypoints to generate
            normalize_actions: If True, normalize joint positions to [-1, 1] range (default True)
            debug: If True, print debug information during solving

        Returns:
            Tuple of (actions_out, q_targets) where:
                - actions_out: List of action tensors for the trajectory
                - q_targets: List of joint target dictionaries
        """
        # Hardcoded controller info for R1Pro robot
        controller_info = {
            'base': {'start_idx': 0, 'dofs': np.array([0, 1, 5]), 'command_dim': 3},
            'trunk': {'start_idx': 3, 'dofs': np.array([6, 7, 8, 9]), 'command_dim': 4},
            'arm_left': {'start_idx': 7, 'dofs': np.array([10, 12, 14, 16, 18, 20, 22]), 'command_dim': 7},
            'gripper_left': {'start_idx': 14, 'dofs': np.array([24, 25]), 'command_dim': 1},
            'arm_right': {'start_idx': 15, 'dofs': np.array([11, 13, 15, 17, 19, 21, 23]), 'command_dim': 7},
            'gripper_right': {'start_idx': 22, 'dofs': np.array([26, 27]), 'command_dim': 1}
        }

        joint_names_use = ['torso_joint1', 'torso_joint2', 'torso_joint3', 'torso_joint4']
        for i in range(1, 8):
            joint_names_use.append(f"left_arm_joint{i}")
        for i in range(1, 8):
            joint_names_use.append(f"right_arm_joint{i}")          

        print("joint_names_use length is", len(joint_names_use)  )

        joint_names = ['torso_joint1', 'torso_joint2', 'torso_joint3', 'torso_joint4', 'left_arm_joint1', 'right_arm_joint1', 'left_arm_joint2', 'right_arm_joint2', 'left_arm_joint3', 'right_arm_joint3', 'left_arm_joint4', 'right_arm_joint4', 'left_arm_joint5', 'right_arm_joint5', 'left_arm_joint6', 'right_arm_joint6', 'left_arm_joint7', 'right_arm_joint7', 'left_gripper_finger_joint1', 'left_gripper_finger_joint2', 'right_gripper_finger_joint1', 'right_gripper_finger_joint2']
        joint_limits = {'torso_joint1': (-1.1344999074935913, 1.8325998783111572), 'torso_joint2': (-2.7924997806549072, 2.5306997299194336), 'torso_joint3': (-1.8325998783111572, 1.5707999467849731), 'torso_joint4': (-3.054299831390381, 3.054299831390381), 'left_arm_joint1': (-4.4506001472473145, 1.309000015258789), 'right_arm_joint1': (-4.4506001472473145, 1.309000015258789), 'left_arm_joint2': (-0.1745000034570694, 3.1415998935699463), 'right_arm_joint2': (-3.1415998935699463, 0.1745000034570694), 'left_arm_joint3': (-2.3561956882476807, 2.3561956882476807), 'right_arm_joint3': (-2.3561956882476807, 2.3561956882476807), 'left_arm_joint4': (-2.0943996906280518, 0.3490999639034271), 'right_arm_joint4': (-2.0943996906280518, 0.3490999639034271), 'left_arm_joint5': (-2.3561956882476807, 2.3561956882476807), 'right_arm_joint5': (-2.3561956882476807, 2.3561956882476807), 'left_arm_joint6': (-1.0471980571746826, 1.0471980571746826), 'right_arm_joint6': (-1.0471980571746826, 1.0471980571746826), 'left_arm_joint7': (-1.5707999467849731, 1.5707999467849731), 'right_arm_joint7': (-1.5707999467849731, 1.5707999467849731), 'left_gripper_finger_joint1': (0.0, 0.05000000074505806), 'left_gripper_finger_joint2': (0.0, 0.05000000074505806), 'right_gripper_finger_joint1': (0.0, 0.05000000074505806), 'right_gripper_finger_joint2': (0.0, 0.05000000074505806)}
        
        if joint_names is not None:
            joint_names = joint_names
        if joint_limits is not None:
            joint_limits = joint_limits

        def parse_target(target_pose_base):
            """Convert target pose tensors to numpy arrays."""
            pos_t, quat_t = target_pose_base
            pos = pos_t.detach().cpu().numpy().astype("float64")
            quat = quat_t.detach().cpu().numpy().astype("float64")
            return pos, quat

        if arms == 'both':

            # Parse goal poses
            pG_left, rG_left = parse_target(left_goal)
            pG_right, rG_right = parse_target(right_goal)

            # Use provided current joint state
            qc = current_joint_positions

            # Solve global IK
            global_ik_config = {
                "q0": qc,
                "pG_left": pG_left,
                "rG_left": rG_left,
                "pG_right": pG_right,
                "rG_right": rG_right,
                "p_err_max": 1e-8,
                "r_err_max": 1e-8,
                "maintain_gaze": 10.0,
            }

            self.global_ik.reset(global_ik_config)
            if self.global_ik.solve():
                if debug:
                    print("Global IK solved")
            else:
                if debug:
                    print("Global IK failed")
                return None, None
                raise RuntimeError("global ik failed!")

            qsol = self.global_ik.get_solution()

        elif arms == 'left':

            pG, rG = parse_target(left_goal)
            qc = current_joint_positions

            config = {
                "pG": pG,
                "rG": rG,
                "q0": np.zeros(len(self.global_single_ik_left.joint_names)),
                "p_err_max": 1e-8,
                "r_err_max": 1e-8,
                "maintain_gaze": 10.0,
            }

            self.global_single_ik_left.reset(config)
            if self.global_single_ik_left.solve():
                if debug:
                    print("Global IK solved")
            else:
                if debug:
                    print("Global IK failed")
                return None, None
                raise RuntimeError("global ik failed!")

            qsol = self.global_single_ik_left.get_solution()            

        elif arms == 'right':

            pG, rG = parse_target(right_goal)
            qc = current_joint_positions

            config = {
                "pG": pG,
                "rG": rG,
                # "q0": qc,
                "q0": np.zeros(len(self.global_single_ik_right.joint_names)),
                "p_err_max": 1e-8,
                "r_err_max": 1e-8,
                "maintain_gaze": 10.0,
            }

            self.global_single_ik_right.reset(config)
            if self.global_single_ik_right.solve():
                if debug:
                    print("Global IK solved")
            else:
                if debug:
                    print("Global IK failed")
                return None, None
                # raise RuntimeError("global ik failed!")

            qsol = self.global_single_ik_right.get_solution()
        
        else:
            raise ValueError("Invalid arms option. Must be 'both', 'left', or 'right'.")

    
        for jindex, jn in enumerate(joint_names_use):
            if jn not in qsol:
                qsol[jn] = current_joint_positions[jindex]

        

        qF = [qsol[n] for n in self.planner.joint_names]
        if debug:
            print("qF", qsol)
            print("++++++++++ SOLVED PLANNER 1 +++++++++++")

        # Generate initial trajectory guess
        t = np.linspace(0, duration, self.T_plan)
        Q0 = np.linspace(qc, qF, self.T_plan)
        dQ0 = np.gradient(Q0, t, axis=0)
        ddQ0 = np.gradient(dQ0, t, axis=0)

        # Configure and solve trajectory planner
        planner_config = {
            "Q0": Q0,
            "dQ0": dQ0,
            "ddQ0": ddQ0,
            "duration": duration,
            "q0": qc,
            "qF": qF,
            "w_dQ": 1.0,
            "w_ddQ": 50.0,
        }

        self.planner.reset(planner_config)

        if self.planner.solve():
            if debug:
                print("[PLANNER] Solved successfully")
        else:
            if debug:
                print("planner failed!")
            raise RuntimeError("planner failed")

        Qsol, _, _ = self.planner.get_solution()

        # Sample trajectory at desired resolution
        t = np.linspace(0, duration, num_targets)
        q_targets = [
            {n: Qsol[n](t[i]) for n in self.planner.joint_names}
            for i in range(num_targets)
        ]
        if debug:
            print(q_targets[-1])
            print("++++++++++ SOLVED PLANNER 2 +++++++++++")

        # Convert joint targets to action sequence
        ctrl_groups = ["trunk", "arm_left", "arm_right"]
        actions_out = [th.zeros(23) for _ in range(num_targets)]

        for i in range(num_targets):
            q_targ = q_targets[i]
            for grp in ctrl_groups:
                if grp not in controller_info:
                    continue

                dof_idx = controller_info[grp]["dofs"]
                start = controller_info[grp]["start_idx"]
                for local_i, j_idx in enumerate(dof_idx.tolist()):
                    j_name = joint_names[j_idx-6]
                    # print(j_name)
                    if j_name in q_targ:
                        if normalize_actions:
                            # Convert absolute joint position to normalized [-1, 1] command
                            normalized_cmd = self.normalize_joint_position(
                                j_name, float(q_targ[j_name]), joint_limits
                            )
                            actions_out[i][start + local_i] = normalized_cmd
                        else:
                            # Use absolute joint position directly
                            actions_out[i][start + local_i] = float(q_targ[j_name])

        # Verify solution with forward kinematics
        qsol = q_targets[-1]
        if debug:
            # print("LEFT GOAL POSITION:", pG_left, rG_left)
            fk = self.ik.left_arm_chain.forward_kinematics(
                {n: qsol[n] for n in self.ik.left_arm_chain.get_joint_parameter_names()}
            )
            print("----------------- LEFT FK POSITION", fk.translation().as_vector())
            print("----------------- LEFT FK Quaternion", fk.rotation().as_quat().toarray().flatten())

            # print("Right GOAL POSITION:", pG_right, rG_right)
            fk = self.ik.right_arm_chain.forward_kinematics(
                {n: qsol[n] for n in self.ik.right_arm_chain.get_joint_parameter_names()}
            )
            print("+++++++++++++++++ Right FK POSITION", fk.translation().as_vector())
            print("+++++++++++++++++ Right FK Quaternion", fk.rotation().as_quat().toarray().flatten())
        return actions_out, q_targets

    def normalize_joint_position(self, joint_name: str, position: float, joint_limits: dict) -> float:
        """Normalize an absolute joint position to [-1, 1] range based on joint limits.

        Args:
            joint_name: Name of the joint
            position: Absolute joint position in radians
            joint_limits: Dictionary mapping joint names to (lower_limit, upper_limit) tuples

        Returns:
            Normalized position in [-1, 1] range
        """
        if joint_name not in joint_limits:
            return 0.0

        lower, upper = joint_limits[joint_name]

        # Map from [lower, upper] to [-1, 1]
        midpoint = (upper + lower) / 2.0
        range_half = (upper - lower) / 2.0

        if range_half == 0:
            return 0.0

        normalized = (position - midpoint) / range_half
        return float(np.clip(normalized, -1.0, 1.0))
