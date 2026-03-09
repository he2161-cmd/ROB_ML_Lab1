import numpy as np
import scipy
np.set_printoptions(precision=3, suppress=True)
from matplotlib import pyplot as plt

def rotation_x(angle):
    return np.array([
        [1, 0, 0, 0],
        [0, np.cos(angle), -np.sin(angle), 0],
        [0, np.sin(angle), np.cos(angle), 0],
        [0, 0, 0, 1]
    ])

def rotation_y(angle):
    return np.array([
        [np.cos(angle), 0, np.sin(angle), 0],
        [0, 1, 0, 0],
        [-np.sin(angle), 0, np.cos(angle), 0],
        [0, 0, 0, 1]
    ])

def rotation_z(angle):
    return np.array([
        [np.cos(angle), -np.sin(angle), 0, 0],
        [np.sin(angle), np.cos(angle), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def translation(x, y, z):
    return np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ])

class InverseKinematics():

    def __init__(self):
        self.joint_positions = None
        self.joint_velocities = None
        self.target_joint_positions = None
        self.counter = 0

        z_stance = -0.14          # ground level
        z_swing  = -0.14 + 0.09  # = -0.05  (top of triangle / mid-swing peak)

        # Six waypoints as required by the assignment:
        touch_down_position  = np.array([ 0.05, 0.0, z_stance])   # front of stance
        stand_position_1     = np.array([ 0.025, 0.0, z_stance])
        stand_position_2     = np.array([ 0.0,  0.0, z_stance])
        stand_position_3     = np.array([-0.025, 0.0, z_stance])
        liftoff_position     = np.array([-0.05, 0.0, z_stance])   # rear of stance
        mid_swing_position   = np.array([ 0.0,  0.0, z_swing])    # top of swing arc

        # ---------------------------
        # Per-leg offsets (base frame)
        # ---------------------------
        rf_ee_offset = np.array([ 0.06, -0.09, 0])
        lf_ee_offset = np.array([ 0.06,  0.09, 0])
        rb_ee_offset = np.array([-0.11, -0.09, 0])
        lb_ee_offset = np.array([-0.11,  0.09, 0])

        # ---------------------------
        # Full waypoint sequences
        # ---------------------------
        # Trotting gait: RF + LB are in phase; LF + RB are offset by half a cycle.
        # RF and LB start at touch_down_position (beginning of stance).
        # LF and RB start at stand_position_3 (middle of stance, already on ground).
        # The sequence closes with touch_down_position repeated so interpolation wraps.

        rf_ee_triangle_positions = np.array([
            ################################################################################################
            # TODO: Implement the trotting gait
            ################################################################################################
            touch_down_position,
            stand_position_1,
            stand_position_2,
            stand_position_3,
            liftoff_position,
            mid_swing_position,
            touch_down_position,   # close the loop
        ]) + rf_ee_offset

        lf_ee_triangle_positions = np.array([
            touch_down_position,
            stand_position_1,
            stand_position_2,
            stand_position_3,
            liftoff_position,
            mid_swing_position,
            touch_down_position,
        ]) + lf_ee_offset

        rb_ee_triangle_positions = np.array([
            touch_down_position,
            stand_position_1,
            stand_position_2,
            stand_position_3,
            liftoff_position,
            mid_swing_position,
            touch_down_position,
        ]) + rb_ee_offset

        lb_ee_triangle_positions = np.array([
            touch_down_position,
            stand_position_1,
            stand_position_2,
            stand_position_3,
            liftoff_position,
            mid_swing_position,
            touch_down_position,
        ]) + lb_ee_offset

        self.ee_triangle_positions = [
            rf_ee_triangle_positions,
            lf_ee_triangle_positions,
            rb_ee_triangle_positions,
            lb_ee_triangle_positions,
        ]
        self.fk_functions = [self.fr_leg_fk, self.fl_leg_fk, self.br_leg_fk, self.bl_leg_fk]

        # self.target_joint_positions_cache, self.target_ee_cache = self.cache_target_joint_positions()
        # print(f'shape of target_joint_positions_cache: {self.target_joint_positions_cache.shape}')
        # print(f'shape of target_ee_cache: {self.target_ee_cache.shape}')


    # -----------------------------------------------------------------------
    # Forward kinematics (one function per leg)
    # -----------------------------------------------------------------------

    def fr_leg_fk(self, theta):
        T_RF_0_1  = translation(0.07500, -0.08350, 0) @ rotation_x(1.57080) @ rotation_z(theta[0])
        T_RF_1_2  = rotation_y(-1.57080) @ rotation_z(theta[1])
        T_RF_2_3  = translation(0, -0.04940, 0.06850) @ rotation_y(1.57080) @ rotation_z(theta[2])
        T_RF_3_ee = translation(0.06231, -0.06216, 0.01800)
        return (T_RF_0_1 @ T_RF_1_2 @ T_RF_2_3 @ T_RF_3_ee)[:3, 3]

    def fl_leg_fk(self, theta):
        T_LF_0_1  = translation(0.07500, 0.08350, 0) @ rotation_x(1.57080) @ rotation_z(-theta[0])
        T_LF_1_2  = rotation_y(-1.57080) @ rotation_z(theta[1])
        T_LF_2_3  = translation(0, -0.04940, 0.06850) @ rotation_y(1.57080) @ rotation_z(-theta[2])
        T_LF_3_ee = translation(0.06231, -0.06216, -0.01800)
        return (T_LF_0_1 @ T_LF_1_2 @ T_LF_2_3 @ T_LF_3_ee)[:3, 3]

    def br_leg_fk(self, theta):
        T_RB_0_1  = translation(-0.07500, -0.07250, 0) @ rotation_x(1.57080) @ rotation_z(theta[0])
        T_RB_1_2  = rotation_y(-1.57080) @ rotation_z(theta[1])
        T_RB_2_3  = translation(0, -0.04940, 0.06850) @ rotation_y(1.57080) @ rotation_z(theta[2])
        T_RB_3_ee = translation(0.06231, -0.06216, 0.01800)
        return (T_RB_0_1 @ T_RB_1_2 @ T_RB_2_3 @ T_RB_3_ee)[:3, 3]

    def bl_leg_fk(self, theta):
        T_LB_0_1  = translation(-0.07500, 0.07250, 0) @ rotation_x(1.57080) @ rotation_z(-theta[0])
        T_LB_1_2  = rotation_y(-1.57080) @ rotation_z(theta[1])
        T_LB_2_3  = translation(0, -0.04940, 0.06850) @ rotation_y(1.57080) @ rotation_z(-theta[2])
        T_LB_3_ee = translation(0.06231, -0.06216, -0.01800)
        return (T_LB_0_1 @ T_LB_1_2 @ T_LB_2_3 @ T_LB_3_ee)[:3, 3]

    def forward_kinematics(self, theta):
        return np.concatenate([self.fk_functions[i](theta[3*i: 3*i+3]) for i in range(4)])

    # -----------------------------------------------------------------------
    # Inverse Kinematics
    # -----------------------------------------------------------------------

    def get_error_leg(self, theta, desired_position):
        """
        Returns a SCALAR representing the Euclidean distance between the
        current end-effector position (given joint angles theta) and the
        desired_position.
        """
        current_position = self.leg_forward_kinematics(theta)
        return np.linalg.norm(current_position - desired_position)

    def inverse_kinematics_single_leg(self, target_ee, leg_index, initial_guess=[0, 0, 0]):
        """
        Returns the joint angles (3,) that place leg `leg_index`'s end-effector
        at target_ee.  Uses scipy.optimize.minimize to minimise the scalar error
        returned by get_error_leg.
        """
        self.leg_forward_kinematics = self.fk_functions[leg_index]
        x0 = np.array(initial_guess, dtype=float)

        result = scipy.optimize.minimize(
            fun=lambda th: self.get_error_leg(th, np.array(target_ee, dtype=float)),
            x0=x0,
            method='L-BFGS-B',
            bounds=[(-np.pi, np.pi)] * 3,
            options={'ftol': 1e-12, 'gtol': 1e-10, 'maxiter': 500},
        )

        return result.x

    # -----------------------------------------------------------------------
    # Trotting gait path interpolation
    # -----------------------------------------------------------------------

    def interpolate_triangle(self, t, leg_index):
        """
        Returns a 3-D position along the triangular foot path for `leg_index`
        at progress t in [0, 1).

        Trotting phase offset:
          RF (0) + LB (3)  -> in phase         (t unchanged)
          LF (1) + RB (2)  -> opposite phase   (t shifted by 0.5)
        """
        if leg_index in (1, 2):
            t = (t + 0.5) % 1.0

        pts = self.ee_triangle_positions[leg_index]
        # pts has 7 rows: [td, s1, s2, s3, lo, ms, td]
        # Stance fraction covers td -> s1 -> s2 -> s3 -> lo  (5 points, 4 segments)
        # Swing  fraction covers lo -> ms -> td               (3 points, 2 segments)

        stance_frac = 0.6   # 60 % of cycle on the ground

        if t < stance_frac:
            u = t / stance_frac          # 0 -> 1 within stance
            n_seg = 4                    # 4 stance segments
            seg = min(int(u * n_seg), n_seg - 1)
            s   = u * n_seg - seg
            a   = pts[seg]
            b   = pts[seg + 1]
        else:
            u   = (t - stance_frac) / (1.0 - stance_frac)  # 0 -> 1 within swing
            n_seg = 2                   # 2 swing segments
            seg = min(int(u * n_seg), n_seg - 1)
            s   = u * n_seg - seg
            # swing points start at index 4 (liftoff)
            a   = pts[4 + seg]
            b   = pts[4 + seg + 1]

        return (1 - s) * a + s * b

    # -----------------------------------------------------------------------
    # Cache
    # -----------------------------------------------------------------------

    def cache_target_joint_positions(self):
        """
        Pre-compute joint angles for every leg over one full gait cycle.
        Returns:
          target_joint_positions_cache : (N, 12) array
          target_ee_cache              : (N, 12) array
        """
        N = 50   # number of time steps per cycle
        t_values = np.linspace(0, 1, N, endpoint=False)

        all_joints = []   # will be (4, N, 3)
        all_ee     = []   # will be (4, N, 3)

        for leg_index in range(4):
            leg_joints = []
            leg_ee     = []
            prev_theta = np.zeros(3)
            for t in t_values:
                target_ee    = self.interpolate_triangle(t, leg_index)
                theta        = self.inverse_kinematics_single_leg(
                                   target_ee, leg_index,
                                   initial_guess=prev_theta)
                prev_theta   = theta
                leg_joints.append(theta)
                leg_ee.append(target_ee)
            all_joints.append(leg_joints)  # (N, 3)
            all_ee.append(leg_ee)          # (N, 3)

        # Stack: (4, N, 3)  ->  reshape to (N, 12)
        all_joints = np.array(all_joints)   # (4, N, 3)
        all_ee     = np.array(all_ee)        # (4, N, 3)

        # Transpose to (N, 4, 3) then reshape to (N, 12)
        target_joint_positions_cache = all_joints.transpose(1, 0, 2).reshape(N, 12)
        target_ee_cache              = all_ee.transpose(1, 0, 2).reshape(N, 12)

        return target_joint_positions_cache, target_ee_cache

    def get_target_joint_positions(self):
        target_joint_positions = self.target_joint_positions_cache[self.counter]
        target_ee              = self.target_ee_cache[self.counter]
        self.counter += 1
        if self.counter >= self.target_joint_positions_cache.shape[0]:
            self.counter = 0
        return target_ee, target_joint_positions


# ---------------------------------------------------------------------------
# Main – validation plots
# ---------------------------------------------------------------------------

def main():

    # Create an instance of the IK class
    inverse_kinematics = InverseKinematics()

    # -----------------------------------------------------------------------
    # Plot 1 – IK validation: target vs achieved EE x-position (front right)
    # -----------------------------------------------------------------------
    target_ee_list = [
        [0.11, -0.09, -0.14],
        [0.10, -0.09, -0.14],
        [0.09, -0.09, -0.14],
        [0.08, -0.09, -0.14],
        [0.07, -0.09, -0.14],
        [0.06, -0.09, -0.14],
        [0.05, -0.09, -0.14],
        [0.04, -0.09, -0.14],
        [0.03, -0.09, -0.14],
        [0.02, -0.09, -0.14],
        [0.01, -0.09, -0.14],
    ]

    inverse_kinematics.leg_forward_kinematics = inverse_kinematics.fk_functions[0]
    result_ee_list = []
    for target_ee in target_ee_list:
        theta     = inverse_kinematics.inverse_kinematics_single_leg(
                        target_ee, leg_index=0, initial_guess=[0, 0, 0])
        result_ee = inverse_kinematics.leg_forward_kinematics(theta)
        result_ee_list.append(result_ee)

    plt.figure()
    plt.plot(np.array(target_ee_list)[:, 0], 'k', label='Target EE Position')
    plt.plot(np.array(result_ee_list)[:, 0], 'ro', label='Result EE Position')
    plt.xlabel('Step')
    plt.ylabel('X (m)')
    plt.legend()
    plt.title('IK Validation – Front Right Leg End Effector X Position')
    plt.tight_layout()
    plt.savefig("lll")

    # Plot the cached trot gait path for one foot.
    # if len(inverse_kinematics.target_ee_cache):
    #     x_list = []
    #     z_list = []
    #     for position in inverse_kinematics.target_ee_cache:
    #         x_list.append(position[0])
    #         z_list.append(position[2])
    #     plt.xlabel('X(m)')
    #     plt.ylabel('Z(m)')
    #     plt.title('EE front right foot trot gait')
    #     plt.plot(x_list, z_list)
    #     plt.show()



if __name__ == '__main__':
    main()