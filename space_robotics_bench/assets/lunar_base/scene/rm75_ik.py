"""Numerical kinematics for the RealMan RM-75 arm on LunarBot (numpy only).

Joint chain copied from openral `packages/rm_description/urdf/rm_75.urdf` (joint origins, axes, limits).
The fixed transform from the rover root frame to the arm `base_link` is not taken from the URDF: callers
calibrate it once in simulation from a measured Link7 pose (`calibrate_mount`), so the model always matches
the articulation that is actually loaded.

The solver is a damped-least-squares IK run on the kinematic model (instantaneous, no dynamics), with joint
limits, a posture bias and several seeds. Its output is a joint-position target for the arm drives; the
executor closes the loop on the residual it measures in simulation.
"""

import math

import numpy as np

# (origin xyz, origin rpy, axis, lower, upper)
CHAIN = (
    ((0, 0, 0.2405), (-1.5708, 0, 0), (0, -1, 0), -3.1067, 3.1067),
    ((0, 0, 0), (1.5708, 0, 0), (0, 1, 0), -2.2689, 2.2689),
    ((0, 0, 0.256), (-1.5708, 0, 0), (0, -1, 0), -3.1067, 3.1067),
    ((0, 0, 0), (1.5708, 0, 0), (0, 1, 0), -2.3562, 2.3562),
    ((0, 0, 0.21), (-1.5708, 0, 0), (0, -1, 0), -3.1067, 3.1067),
    ((0, 0, 0), (1.5708, 0, 0), (0, 1, 0), -2.234, 2.234),
    ((0, 0, 0.144), (0, 0, 0), (0, 0, 1), -6.2832, 6.2832),
)
Q_LO = np.array([c[3] for c in CHAIN]) + 0.05
Q_HI = np.array([c[4] for c in CHAIN]) - 0.05


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _axis_angle(a, q):
    a = np.asarray(a, float)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(q) * K + (1 - math.cos(q)) * K @ K


_ORIGINS = []
for xyz, rpy, _, _, _ in CHAIN:
    T = np.eye(4)
    T[:3, :3] = _rpy(*rpy)
    T[:3, 3] = xyz
    _ORIGINS.append(T)


def fk(q, mount=np.eye(4), tcp=(0.0, 0.0, 0.14)):
    """Return (tcp position, Link7 rotation, per-joint (origin, world axis)) in the mount's parent frame."""
    T = mount.copy()
    joints = []
    for i, (_, _, axis, _, _) in enumerate(CHAIN):
        T = T @ _ORIGINS[i]
        joints.append((T[:3, 3].copy(), T[:3, :3] @ np.asarray(axis, float)))
        R = np.eye(4)
        R[:3, :3] = _axis_angle(axis, q[i])
        T = T @ R
    return T[:3, 3] + T[:3, :3] @ np.asarray(tcp), T[:3, :3], joints, T


def jacobian(q, mount, tcp=(0.0, 0.0, 0.14)):
    p, R, joints, _ = fk(q, mount, tcp)
    J = np.zeros((6, 7))
    for i, (o, a) in enumerate(joints):
        J[:3, i] = np.cross(a, p - o)
        J[3:, i] = a
    return J, p, R


def calibrate_mount(q, link7_pos, link7_rot):
    """Fixed transform root->arm base_link from one measured Link7 pose (both expressed in the root frame)."""
    _, _, _, T7 = fk(q)  # base_link -> Link7
    Tm = np.eye(4)
    Tm[:3, :3] = link7_rot
    Tm[:3, 3] = link7_pos
    return Tm @ np.linalg.inv(T7)


def rot_err(R_t, R):
    Re = R_t @ R.T
    ang = math.acos(max(-1.0, min(1.0, (np.trace(Re) - 1) / 2)))
    if ang < 1e-6:
        return np.zeros(3)
    if ang > math.pi - 1e-3:  # axis ill-defined at 180 deg; any perpendicular kick works
        w, v = np.linalg.eigh((Re + Re.T) / 2)
        return v[:, -1] * ang
    axis = np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]]) / (2 * math.sin(ang))
    return axis * ang


def solve(p_t, mount, R_t=None, z_t=None, seeds=(), posture=None, iters=300, tol=1e-3, rot_tol=0.02, n_random=8):
    """IK for the TCP. Give either a full tool rotation `R_t` or only a tool approach axis `z_t` (roll free).

    Returns (q, pos_err, rot_err) of the best solution over the seeds (lowest error, then closest to seed 0).
    """
    p_t = np.asarray(p_t, float)
    posture = np.zeros(7) if posture is None else np.asarray(posture, float)
    best = None
    rng = np.random.default_rng(0)
    seeds = list(seeds) + [rng.uniform(Q_LO, Q_HI) * np.array([1, 0.8, 1, 0.8, 1, 0.8, 0.3]) for _ in range(n_random)]
    for s in seeds:
        q = np.clip(np.asarray(s, float), Q_LO, Q_HI)
        for _ in range(iters):
            J, p, R = jacobian(q, mount)
            e_p = p_t - p
            if R_t is not None:
                e_r = rot_err(R_t, R)
                Jr = J
            else:  # align tool z with z_t; rotation about the tool axis is unconstrained
                e_r = np.cross(R[:, 2], z_t)
                Jr = J
            ep_n, er_n = np.linalg.norm(e_p), np.linalg.norm(e_r)
            if ep_n < tol and er_n < rot_tol:
                break
            err = np.concatenate([e_p if ep_n < 0.1 else e_p / ep_n * 0.1, e_r if er_n < 0.3 else e_r / er_n * 0.3])
            if R_t is None:  # only the two rotation directions perpendicular to the tool axis are constrained
                UV = R[:, :2].T
                A = np.vstack([Jr[:3], UV @ Jr[3:]])
                err = np.concatenate([err[:3], UV @ err[3:]])
            else:
                A = Jr
            free = np.ones(7, bool)
            for _ in range(7):  # joints sitting on a limit and pushed outward drop out of the solve
                W = np.diag(free.astype(float))
                AW = A @ W
                Ainv = W @ AW.T @ np.linalg.inv(AW @ AW.T + 0.02 ** 2 * np.eye(len(A)))
                dq = Ainv @ err + 0.05 * (np.eye(7) - Ainv @ A) @ W @ (posture - q)
                out = ((q <= Q_LO + 1e-4) & (dq < 0)) | ((q >= Q_HI - 1e-4) & (dq > 0))
                if not (out & free).any():
                    break
                free &= ~out
            q = np.clip(q + np.clip(dq, -0.2, 0.2), Q_LO, Q_HI)
        p, R, _, _ = fk(q, mount)
        pe = float(np.linalg.norm(p_t - p))
        re = float(np.linalg.norm(rot_err(R_t, R))) if R_t is not None else float(np.linalg.norm(np.cross(R[:, 2], z_t)))
        score = (round(pe, 3) + 0.2 * round(re, 2), float(np.abs(q - np.asarray(seeds[0])).sum()))
        if best is None or score < best[0]:
            best = (score, q, pe, re)
    return best[1], best[2], best[3]
