import json
from utils.pose_util import *
from myIK import MyIK

def play_traj(poses):
    joint_angles = [-1.57, -1.57, 1.57, -1.57, -1.57, 0]
    myIK = MyIK(use_ikfast=False)
    init_pose = myIK.forward_joints(joint_angles)
    print('init_pose', init_pose)

    p_star_list = []
    for p in poses:
        R, t = pose_to_Rt(p)
        # print(R)
        p_star = transform_pose(R, t, init_pose)
        p_star_list.append(p_star)

    visualize_poses(p_star_list, label="points to plan", color='y', autoscale=False, ax=None)
    plt.show()
    traj = myIK.plan_trajectory(p_star_list, joint_angles)
    myIK.show_traj(traj,loop=False)


def show_poses_traj():
    with open("poses_traj.json", "r") as json_file:
        traj = json.load(json_file)
        print(traj.keys())
        i = '0'
        color_map = {'0':'b',
        '1':'r',
        '2':'g',
        '4':'y',
        }
        ax = None
        for i in ['0', '1',  '2', '4']:
            poses = traj[i][:]
            ax = visualize_poses(poses, label=i, color=color_map[i],  ax=ax,   autoscale=True)
        plt.show()


def get_object_poses(filename):
    with open(filename, "r") as json_file:
        traj = json.load(json_file)
        print(traj.keys())
        i = '1'
        # play the traj to robot
        object_poses = traj[i]
        # take the first as reference
        return object_poses
    return None


def reverse_poses(poses):
    p_star_list = []
    for p in poses:
        t = np.array(p[:3])
        R = quat_to_R(np.array(p[3:]))
        t_star = -R.T @ t
        R_star = R.T
        p_star = list(t_star) + list(R_to_quat(R_star))
        p_star_list.append(p_star)
    return p_star_list


def pose_delta(pose0, pose1):
    R0 = quat_to_R(pose0[3:])
    R1 = quat_to_R(pose1[3:])
    R_diff =  R0 @ R1.T # equal to np.dot(R0, R1.T) but R0 * R1  is elementwise multiply
    t_diff =  np.array(pose0[:3]) - R_diff @ np.array(pose1[:3])
    return np.array(list(t_diff)+list(R_to_quat(R_diff)))


if __name__=="__main__":
    object_poses = get_object_poses(filename="data/camera.json")

    camera_poses = reverse_poses(object_poses)
    reference_pose = camera_poses[0]

    ax = visualize_poses(camera_poses, label='camera', color='b', autoscale=False)
    ax = visualize_poses(object_poses, label='object', color='g', ax=ax)

    object_poses_delta = object_poses
    object_poses_delta = [pose_delta(p, reference_pose) for p in object_poses_delta]
    ax = visualize_poses(object_poses_delta, label='delta', color='y', ax=ax)
    plt.show()
    # # show_poses_traj()
    play_traj(object_poses_delta)