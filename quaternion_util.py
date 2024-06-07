from scipy.spatial.transform import Rotation

def relative_rotation(q1, q2):
    """
    Compute the relative rotation from quaternion q1 to quaternion q2.

    Args:
    - q1: Quaternion representing the initial rotation
    - q2: Quaternion representing the final rotation

    Returns:
    - relative_quaternion: Quaternion representing the relative rotation from q1 to q2
    """
    # Convert quaternions to rotation objects
    rot1 = Rotation.from_quat(q1)
    rot2 = Rotation.from_quat(q2)

    # Compute the relative rotation from rot1 to rot2
    relative_rotation = rot2 * rot1.inv()

    return relative_rotation.as_matrix()

if __name__=="__main__":
    # Example quaternions (w, x, y, z)
    q1 = [0.707, 0.0, 0.707, 0.0]  # Example quaternion 1
    q2 = [0.5, 0.5, 0.5, 0.5]      # Example quaternion 2

    R = relative_rotation(q1, q2)
    print("Relative rotation quaternion:", R)

    from pose_util import *
    q2_star = R_dot_quat(R, q1)
    print(q2_star, q2)