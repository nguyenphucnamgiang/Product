# Standard Library imports
# import sys

# Third-party imports
import torch
import numpy as np
import open3d as o3d
import cv2

# Local application/library imports
# sys.path.append(r'C:\Users\namgi\Nerfstudio\Depth_Anything\Depth-Anything-V2')
from depth_anything_v2.dpt___original import DepthAnythingV2


# calculate camera matrix
def get_intrinsics (H,W, fov=55.0) :
    """
    Intrinsics for a pinhole camera model.
    Assume fov of 55 degrees and central principal point. """
    f = 0.5 * W/ np.tan (0.5 * fov * np.pi / 180.0)
    cx = 0.5 * W
    cy = 0.5 * H
    return np.array([[f, 0, cx],
                    [0, f, cy],
                    [0, 0, 1]])

def use_model(input_encoder):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}   # UNAVAILABLE
    }

    encoder =  input_encoder # or 'vits', 'vitb', 'vitl', 'vitg'

    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(
        fr"checkpoints\depth_anything_v2_{encoder}.pth",
        map_location='cpu',
        weights_only=False))
    model = model.to(DEVICE).eval()
    return model


def prepare_data(image_path, model):
    try:
        raw_img = cv2.imread(image_path)
        color_image = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        FINAL_HEIGHT, FINAL_WIDTH, channels = color_image.shape

        pred = model.infer_image(color_image)

        gray_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
        ret, raw_mask = cv2.threshold(gray_img, 10, 255, cv2.THRESH_BINARY)

        kernel = np.ones((10, 10), np.uint8)
        mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)

        # Contour-based hole filling
        contours, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        cv2.drawContours(image=mask, contours=contours, contourIdx=-1, color=(255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)

        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        resized_pred = cv2.resize(pred, (FINAL_WIDTH, FINAL_HEIGHT), interpolation=cv2.INTER_NEAREST)
        resized_mask = cv2.resize(mask, (FINAL_WIDTH, FINAL_HEIGHT), interpolation=cv2.INTER_NEAREST)

        return  FINAL_WIDTH, FINAL_HEIGHT, resized_pred, resized_mask, color_image,

    except Exception as e:
        print(f"Error processing {image_path}: {type(e).__name__} - {e}")


def mask_and_remove_points(FINAL_WIDTH, FINAL_HEIGHT, resized_pred, resized_mask, color_image):
    camera_matrix = get_intrinsics(FINAL_HEIGHT, FINAL_WIDTH)
    FX = camera_matrix[0, 0]
    FY = camera_matrix[1, 1]
    FL = (FX + FY) / 2

    focal_length_x, focal_length_y = (FX, FY)
    # Timer for generating the 3D point cloud
    x, y = np.meshgrid(np.arange(FINAL_WIDTH), np.arange(FINAL_HEIGHT))
    z = np.array(resized_pred)

    x1 = (x - FINAL_WIDTH / 2) / focal_length_x
    y1 = (y - FINAL_HEIGHT / 2) / focal_length_y

    points = np.stack((-np.multiply(x1, z), np.multiply(y1, z), z), axis=-1).reshape(-1, 3)

    colors = np.array(color_image).reshape(-1, 3) / 255.0

    filter_mask = np.array(resized_mask).reshape(-1) > 0

    points_masked = points[filter_mask]
    colors_masked = colors[filter_mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_masked)
    pcd.colors = o3d.utility.Vector3dVector(colors_masked)

    o3d.visualization.draw_geometries([pcd], )

def main():
    # mask and remove points using active contours
    image_path = input("Please enter the full path of the image: ")


    encoder = 'vitl'
    model = use_model(encoder)

    FINAL_WIDTH, FINAL_HEIGHT, resized_pred, resized_mask, color_image = prepare_data(image_path, model)

    mask_and_remove_points(FINAL_WIDTH, FINAL_HEIGHT, resized_pred, resized_mask, color_image)


if __name__ == '__main__':
    main()
