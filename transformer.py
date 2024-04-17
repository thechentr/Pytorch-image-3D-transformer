import torch
from torchvision import transforms
from torchvision.transforms import functional as F

import math
import time
from PIL import Image

def save_tensor_image(image_tensor, file_path):
    """
    Saves an image tensor to a specified path, supporting tensors with value ranges [0, 1].
    
    Parameters:
        image_tensor (torch.Tensor): The image tensor to be saved, with shape (C, H, W) or (N, C, H, W).
        file_path (str): The path where the image will be saved.
    """
    # Handle batch dimension if present
    if image_tensor.ndim == 4 and image_tensor.shape[0] == 1:
        image_tensor = image_tensor.squeeze(0)
        
    if image_tensor.ndim != 3 or image_tensor.shape[0] not in [1, 3]:
        raise ValueError("Image tensor shape must be (C, H, W), where C is 1 or 3")

    # Adjust the tensor values if necessary
    min_val = image_tensor.min()
    max_val = image_tensor.max()
    if min_val >= 0 and max_val <= 1:
        image_tensor = (image_tensor * 255).clamp(0, 255).byte()
    else:
        err = f"Unsupported image tensor value range {min_val}-{max_val}"
        raise ValueError(err)

    # Convert to PIL Image and save
    if image_tensor.shape[0] == 3:
        image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        image = Image.fromarray(image_np, 'RGB')
    elif image_tensor.shape[0] == 1:
        image_np = image_tensor.squeeze(0).cpu().numpy()
        image = Image.fromarray(image_np, 'L')
        
    image.save(file_path)

def _create_camera_matrix(fov):
    """
    Creates a camera intrinsic matrix based on a given field of view (FOV).

    Parameters:
        fov (float): Field of view in degrees.

    Returns:
        torch.Tensor: The camera matrix.
    """
    # Convert FOV from degrees to radians
    fov_rad = math.radians(fov)

    # Compute the focal length
    f_x = 1 / (torch.tan(torch.tensor(fov_rad / 2)) * 1.4142)  # sqrt(2) = 1.4142
    f_y = 1 / (torch.tan(torch.tensor(fov_rad / 2)) * 1.4142)

    # Principal point coordinates (at the center of the normalized image)
    c_x = 0.5
    c_y = 0.5

    # Construct the intrinsic matrix
    K = torch.tensor([
        [f_x, 0, c_x],
        [0, f_y, c_y],
        [0, 0, 1]
    ])

    # Extend to 4x4
    camera_matrix = torch.eye(4)
    camera_matrix[:3, :3] = K

    return camera_matrix

def world2img(p, matrix):
    """
    Projects a 3D point in world coordinates to 2D image coordinates using a projection matrix.

    Parameters:
        p (torch.Tensor): A 4D point in homogeneous coordinates.
        matrix (torch.Tensor): The projection matrix.

    Returns:
        list: 2D coordinates in image space.
    """
    p = torch.matmul(matrix, p)  # Apply camera intrinsics
    p = p / p[2]  # Normalize by the z-component
    return [float(p[0]), float(p[1])]

def _pad_with_black(input_tensor, target_size):
    """
    Pads an input tensor to a specified size with black borders.
    
    Parameters:
        input_tensor (torch.Tensor): The input tensor, size [N, C, H, W].
        target_size (tuple): The target size as (target_height, target_width).
    
    Returns:
        torch.Tensor: The padded tensor, size [N, C, target_height, target_width].
    """
    _, input_height, input_width = input_tensor.shape[-3:]
    target_height, target_width = target_size
    
    pad_vertical = max(target_height - input_height, 0)
    pad_horizontal = max(target_width - input_width, 0)
    
    pad_top_bottom = pad_vertical // 2
    pad_left_right = pad_horizontal // 2
    
    padded_tensor = F.pad(input_tensor, (pad_left_right, pad_top_bottom), fill=0)
    
    return padded_tensor

def _rotate_about_axis(points, angle_rad, axis):
    """
    Rotates 3D points about a specified axis by a given angle.

    Parameters:
        points (list): List of 3D points.
        angle_rad (float): Rotation angle in radians.
        axis (str): Axis of rotation ('x', 'y', or 'z').

    Returns:
        list: Transformed points.
    """
    angle_rad = angle_rad + 1e-5  # Prevent division by zero in perspective transformations
    cos, sin = math.cos(angle_rad), math.sin(angle_rad)
    
    if axis == 'x':
        R = torch.tensor([[1, 0, 0, 0], [0, cos, -sin, 0], [0, sin, cos, 0], [0, 0, 0, 1]], dtype=torch.float)
    elif axis == 'y':
        R = torch.tensor([[cos, 0, sin, 0], [0, 1, 0, 0], [-sin, 0, cos, 0], [0, 0, 0, 1]], dtype=torch.float)
    elif axis == 'z':
        R = torch.tensor([[cos, -sin, 0, 0], [sin, cos, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float)
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'.")

    bais = torch.tensor([0., 0., 4., 0.])
    center = sum(points) / len(points) + bais
    T_neg = torch.eye(4)
    T_neg[:3, 3] = -center[:3]
    T_pos = torch.eye(4)
    T_pos[:3, 3] = center[:3]

    # Combine translation and rotation into a single transformation matrix
    K = torch.matmul(T_pos, torch.matmul(R, T_neg))
    return [K @ p for p in points]

def rotate(img_tensor, horizontal, vertical, canvas_size=(128, 128), radius=300, FOV=60, interpolation=transforms.InterpolationMode.BILINEAR):
    """
    Applies a 3D rotation to an image tensor based on specified horizontal and vertical angles.
    
    Parameters:
        img_tensor (torch.Tensor): Image tensor to rotate.
        horizontal (float): Horizontal rotation angle in radians.
        vertical (float): Vertical rotation angle in radians.
        canvas_size (tuple): Size of the output canvas.
        radius (int): Radius of the spherical projection.
        FOV (int): Field of view in degrees.
        interpolation (transforms.InterpolationMode): Interpolation method for transformation.
    
    Returns:
        tuple: Transformed image tensor and mask.
    """
    device = img_tensor.device
    mask = torch.ones_like(img_tensor, device=device)
    
    img_tensor = _pad_with_black(img_tensor, canvas_size)
    mask = _pad_with_black(mask, canvas_size)

    canvas_h, canvas_w = canvas_size

    points_orig = [[0, 0], [0, canvas_w], [canvas_h, canvas_w], [canvas_h, 0]]

    # Initialize points
    w, h = img_tensor.shape[1:]
    points_world = [
        torch.tensor([-w/2, -h/2, radius, 1.]),
        torch.tensor([-w/2, h/2, radius, 1.]),
        torch.tensor([w/2, h/2, radius, 1.]),
        torch.tensor([w/2, -h/2, radius, 1.])
    ]
    
    camera_matrix = _create_camera_matrix(FOV)
    rotated_points_world = _rotate_about_axis(points_world, horizontal, 'y')  # Rotate and project
    rotated_points_world = _rotate_about_axis(rotated_points_world, vertical, 'x')

    antinorm = lambda x: (x[0]*canvas_w, x[1]*canvas_h)
    rotated_points = [antinorm(world2img(p, camera_matrix)) for p in rotated_points_world]

    transformed_img_tensor = F.perspective(img_tensor, points_orig, rotated_points, interpolation=interpolation, fill=0)
    transformed_mask = F.perspective(mask, points_orig, rotated_points, interpolation=interpolation, fill=0)

    return transformed_img_tensor, transformed_mask

if __name__ == '__main__':

    tensor_image = torch.rand((3, 32, 64))

    i = 0
    while True:
        i += 0.2
        transformed_image, transformed_mask = rotate(tensor_image, horizontal=i, vertical=0, canvas_size=(128, 128))
        save_tensor_image(transformed_image, f'transformed_image.png')
        save_tensor_image(transformed_mask, f'transformed_mask.png')
        time.sleep(0.2)
