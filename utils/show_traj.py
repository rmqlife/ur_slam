import json
from pose_util import *

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

show_poses_traj()