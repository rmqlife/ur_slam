import click
from utils.aruco_util import *
from utils.lightglue_util import load_intrinsics

@click.command()
@click.option("--image-path", type=str, required=True, default="/home/rmqlife/work/ur_slam/utils/aruco_test.jpg", help="Path to the image containing ArUco markers")
def main(image_path):
    # Load the image
    image = cv2.imread(image_path)
    intrinsics = load_intrinsics("slam_data/intrinsics_d435.json")
    print(image.shape)
    # Call the function to detect ArUco markers
    corners, ids = detect_aruco(image, draw_flag=True)
    if ids is not None and len(ids) > 0:
            print(f'corner points: {corners}')
            print(f'ids: {ids}')
            poses = estimate_markers_poses(corners, marker_size=0.03, intrinsics=intrinsics)
            print('poses', poses)
    else:
        print('Image not found or unable to load.')
    cv2.imshow('main', image)
    cv2.waitKey(0)

# Example usage
def main_gen_aruco():
    marker_size = 100  # Size of the marker image in pixels
    for marker_id in range(20):
        output_file = f'arucos/aruco_marker_{marker_id}.png'  # Output file name
        generate_aruco_marker(marker_id, marker_size, output_file)

if __name__=="__main__":
     main()