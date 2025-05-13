import numpy as np
from scipy.ndimage import affine_transform,gaussian_filter
import random
# from skimage import exposure
import random

def generate_rotation_matrix(angles, order='xyz'):
    """
    Generate a 3D rotation transformation matrix by combining rotations 
    around the x, y, and z axes.

    Parameters:
    -----------
    angles : tuple of floats
        Rotation angles (in radians) around the x, y, and z axes, in the order specified.
        Example: (angle_x, angle_y, angle_z)
    
    order : str
        Order of rotations, specified as a string of 'x', 'y', and 'z'. Default is 'xyz'.
        Example: 'xyz', 'zyx', 'yxz', etc.

    Returns:
    --------
    numpy.ndarray
        A 3x3 rotation matrix representing the combined rotation.
    """
    if len(angles) != 3 or len(order) != 3:
        raise ValueError("Both 'angles' and 'order' must have exactly three elements.")
    
    # Rotation matrices for basic axes
    def Rx(theta):
        return np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])
    
    def Ry(theta):
        return np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
    
    def Rz(theta):
        return np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
    
    # Map axis names to rotation functions
    axis_map = {'x': Rx, 'y': Ry, 'z': Rz}
    
    # Start with an identity matrix
    rotation_matrix = np.eye(3)
    
    # Apply rotations in the specified order
    for axis, angle in zip(order, angles):
        rotation_matrix = rotation_matrix @ axis_map[axis](angle)
    
    return rotation_matrix

def rotate_volume(volume, rotation_matrix):
    """
    Apply a 3D rotation to a volume using a given rotation matrix.

    Parameters:
    -----------
    volume : numpy.ndarray
        3D image volume to be rotated (e.g., shape [128, 128, 128]).
    
    rotation_matrix : numpy.ndarray
        A 3x3 rotation matrix to apply.

    Returns:
    --------
    numpy.ndarray
        The rotated volume.
    """
    # Compute the center of the volume
    center = np.array(volume.shape) / 2

    # Define the full affine transformation matrix (4x4)
    # Add translation to rotate around the center of the volume
    affine_matrix = np.eye(4)
    affine_matrix[:3, :3] = rotation_matrix  # Insert the rotation part
    translation = center - rotation_matrix @ center
    affine_matrix[:3, 3] = translation  # Add translation to align the rotation around the center

    # Apply the affine transformation
    rotated_volume = affine_transform(
        volume,
        matrix=rotation_matrix,
        offset=translation,
        order=3,  # Cubic interpolation
        mode='constant',  # Fill with 0s outside the volume
        cval=0.0
    )
    return rotated_volume

def generate_random_angles(lower_limit, upper_limit):
    """
    Generate three random rotation angles within the specified range.

    Parameters:
    -----------
    lower_limit : float
        The lower limit of the angle range (in radians).
    upper_limit : float
        The upper limit of the angle range (in radians).

    Returns:
    --------
    tuple of floats
        Three random rotation angles (angle_x, angle_y, angle_z) in radians.
    
    Example:
    --------
    >>> lower_limit = -np.pi / 4  # -45 degrees in radians
    >>> upper_limit = np.pi / 4   # 45 degrees in radians
    >>> random_angles = generate_random_angles(lower_limit, upper_limit)
    >>> print("Random Rotation Angles (in radians):", random_angles)
    """
    if lower_limit > upper_limit:
        raise ValueError("Lower limit must be less than or equal to the upper limit.")
    
    # Generate three random angles uniformly distributed in the specified range
    angles = np.random.uniform(lower_limit, upper_limit, size=3)
    return tuple(angles)

def generate_3d_volume_with_line(volume_shape, thickness, line_value, background_value):
    """
    Generate a 3D volume with a line stretching from the top-left to the bottom-right.

    Parameters:
    -----------
    volume_shape : tuple of int
        Shape of the 3D volume (depth, height, width).
    thickness : int
        Thickness of the line in pixels.
    line_value : int or float
        Value assigned to the pixels of the line.
    background_value : int or float
        Value assigned to the background pixels.

    Returns:
    --------
    np.ndarray
        3D volume with the specified line.
    """
    volume = np.full(volume_shape, background_value, dtype=np.float32)
    depth, height, width = volume_shape

    for z in range(depth):
        # Calculate the center coordinates of the line at each depth
        y = int(height * z / depth)
        x = int(width * z / depth)
        
        # Define the line thickness (in a square cross-section)
        for dy in range(-thickness // 2, thickness // 2 + 1):
            for dx in range(-thickness // 2, thickness // 2 + 1):
                yy = np.clip(y + dy, 0, height - 1)
                xx = np.clip(x + dx, 0, width - 1)
                volume[z, yy, xx] = line_value

    return volume

def random_rotation_3d(volume,lower_limit = -np.pi/6,upper_limit = np.pi/6):
    angles = generate_random_angles(lower_limit,upper_limit)
    rotation_matrix = generate_rotation_matrix(angles,order ='zyx')
    rotated_volume = rotated_volume(volume,rotation_matrix)
    return rotated_volume


    

def center_crop_3d(volume, crop_shape):
    """
    Perform a center crop on a 3D volume.

    Parameters:
    -----------
    volume : np.ndarray
        The 3D volume to be cropped, with shape (depth, height, width).
    crop_shape : tuple of int
        The desired crop shape (crop_depth, crop_height, crop_width).

    Returns:
    --------
    np.ndarray
        The cropped 3D volume.
    """
    if len(volume.shape) != 3:
        raise ValueError("Input volume must be a 3D array.")

    if any(cs > vs for cs, vs in zip(crop_shape, volume.shape)):
        raise ValueError("Crop shape must not exceed the dimensions of the input volume.")

    depth, height, width = volume.shape
    crop_depth, crop_height, crop_width = crop_shape

    # Calculate the start and end indices for each dimension
    start_d = (depth - crop_depth) // 2
    start_h = (height - crop_height) // 2
    start_w = (width - crop_width) // 2

    end_d = start_d + crop_depth
    end_h = start_h + crop_height
    end_w = start_w + crop_width

    # Perform the crop
    cropped_volume = volume[start_d:end_d, start_h:end_h, start_w:end_w]
    return cropped_volume

def random_gaussian_blur_3d(volume,P=0.8,sigma_range=(0.1, 2)):
    """
    Apply a random Gaussian blur to a 3D volume.
    
    Parameters:
    - volume (numpy.ndarray): Input 3D volume to be blurred.
    - sigma_range (tuple): Range of sigma values for the Gaussian blur.
    
    Returns:
    - blurred_volume (numpy.ndarray): The blurred 3D volume.
    """

    p = random.random()
    if p < P:
        # Generate random sigma within the specified range
        sigma = np.random.uniform(sigma_range[0], sigma_range[1])
        print(f"sigma of blur is {sigma}")
        
        # Apply Gaussian filter
        blurred_volume = gaussian_filter(volume, sigma=sigma)
        return blurred_volume

    else :
        return volume




def random_brightness_3d(image, P=0.8,clip = True,brightness_factor_range=(0.9, 1.1), contrast_s =0.05, low_thres= 96,high_noise_threshold=4000):
    """
    Adjusts brightness and contrast of a 3D grayscale image with probability P
        
    Returns:
        ndarray: Adjusted 3D image.
    """
    # Step 1: Clip the values above the noise threshold (99th percentile)
    
    # Step 2: Adjust Brightness
    p = random.random()
    if p < P:
        brightness_factor = random.uniform(*brightness_factor_range)
        print(f"brightness_factor is {brightness_factor}")
        if clip:
            image_bright = np.clip(image* brightness_factor, low_thres, high_noise_threshold)  # Ensure the values stay within the valid range
        else:
            image_bright = image * brightness_factor

        return image_bright
        
        # did not find a suitable contrast adjust strategy
        # # Step 3: Adjust Contrast
        # image_range = [np.min(image_bright),np.max(image_bright)]
        # contrast_factor = random.uniform(0,contrast_s)
        # print(f"contrsat_factor is {contrast_factor}")
        # out_range = ( (1-contrast_factor) * image_range[0],(1+contrast_factor) * image_range[1])
        # print(f"ori_range {image_range},out_range{out_range}")
        # image_contrast = exposure.rescale_intensity(image_bright,  out_range=out_range)     
        # return image_contrast
    else:
        return image
class RandomRotation3D:
    def __init__(self, lower_limit=-np.pi/6, upper_limit=np.pi/6,probability=0.8,v=False):
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.probability = probability
        self.v = v

    def __call__(self, volume):
        if random.random() < self.probability:
            angles = generate_random_angles(self.lower_limit, self.upper_limit)
            rotation_matrix = generate_rotation_matrix(angles, order='zyx')
            rotated_volume = rotate_volume(volume, rotation_matrix)
            return rotated_volume
        if self.v:
            print(f"this time did not rotated")
        return volume
        

class CenterCrop3D:
    def __init__(self, crop_shape):
        self.crop_shape = crop_shape

    def __call__(self, volume):
        if len(volume.shape) != 3:
            raise ValueError("Input volume must be a 3D array.")

        if any(cs > vs for cs, vs in zip(self.crop_shape, volume.shape)):
            raise ValueError("Crop shape must not exceed the dimensions of the input volume.")

        depth, height, width = volume.shape
        crop_depth, crop_height, crop_width = self.crop_shape

        start_d = (depth - crop_depth) // 2
        start_h = (height - crop_height) // 2
        start_w = (width - crop_width) // 2

        end_d = start_d + crop_depth
        end_h = start_h + crop_height
        end_w = start_w + crop_width

        return volume[start_d:end_d, start_h:end_h, start_w:end_w]




class RandomGaussianBlur3D:
    def __init__(self, sigma_range=(0.1, 2), probability=0.8,v=False):
        self.sigma_range = sigma_range
        self.probability = probability
        self.v = v

    def __call__(self, volume):
        if random.random() < self.probability:
            sigma = np.random.uniform(*self.sigma_range)
            if self.v:
                print(f"sigma of blur is {sigma}")
            return gaussian_filter(volume, sigma=sigma)
        if self.v:
            print(f"this time did not blur")
        return volume
    
class RandomBrightness3D:
    def __init__(self, brightness_factor_range=(0.9, 1.1), clip=True, probability=0.8, 
                 contrast_s=0.05, low_thres=96, high_noise_threshold=4000,v=False):
        self.brightness_factor_range = brightness_factor_range
        self.clip = clip
        self.probability = probability
        self.contrast_s = contrast_s
        self.low_thres = low_thres
        self.high_noise_threshold = high_noise_threshold
        self.v = v

    def __call__(self, image):
        if random.random() < self.probability:
            brightness_factor = random.uniform(*self.brightness_factor_range)
            if self.v:
                print(f"brightness_factor is {brightness_factor}")
            if self.clip:
                return np.clip(image * brightness_factor, self.low_thres, self.high_noise_threshold)
            return image * brightness_factor
        if self.v:
            print(f"this time did not adjust brightness")
        return image


def apply_transforms(data, transforms):
    for transform in transforms:
        data = transform(data)
    return data

