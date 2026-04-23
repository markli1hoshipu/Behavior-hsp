from mark_ik import solve_ik

import numpy as np

def main():
    # Local configuration (no CLI)
    mode = "left_torso"  # or "right_torso"
    # Use packaged URDF by default; or pass a custom path
    urdf_path = None
    # Extra tool link appended after URDF EEF (set exactly one or leave both None)
    extra_link_xyzquat = [0.0, 0.0, -0.06, 0.0, 1.0, 0.0, 0.0]

    # Target pose to solve for (example)
    target_pos_m = np.array([0.13937175273895264,
                            0.2525796890258789,
                            0.43750888109207153
                            ], dtype=np.float64)
    target_quat_xyzw = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)

    initial_guess = np.array([        
         1.0250016450881958,
        -1.4499949216842651,
        -0.4699985384941101,
        0.000010797142749652267,     # torso_joint1..4
        0.0015461769653484225,
        0.0015347963199019432,
        0.0018061291193589568,
        -0.0353488065302372,
        0.004601459950208664,
        -0.010740505531430244,
        -0.003066019155085087,  # arm_joint1..7
    ], dtype=np.float64)
        
    # Early stop tolerances
    pos_tol = 0.01  # ~1cm
    ori_tol = 0.1   # ~5deg

    res = solve_ik(
        mode,
        target_pos_m,
        target_quat_xyzw,
        urdf_path=urdf_path,
        extra_link_xyzquat=extra_link_xyzquat,
        initial_guess=initial_guess,
        max_iters=1000,
        lr=1e-3,
        pos_weight=10.0,
        ori_weight=1.0,
        pos_tol=pos_tol,
        ori_tol=ori_tol,
        verbose=True,
    )
    print("IK solution (11 joints):", res["q_sol"].astype(np.float64))
    print("Final objective:", res["best_loss"])
    print("Target pos:", res["target_pos"].astype(np.float64))
    print("Target quat_xyzw:", res["target_quat"].astype(np.float64))
    print("Solved pos:", res["solved_pos"].astype(np.float64))
    print("Solved quat_xyzw:", res["solved_quat"].astype(np.float64))
    print("Errors -> pos (m):", res["err_pos"], ", ori (rad):", res["err_quat"])


if __name__ == "__main__":
    main()

