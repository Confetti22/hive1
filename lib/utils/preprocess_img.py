import numpy as np
def pad_to_multiple_of_unit(img,unit = 8):
    H, W = img.shape[-2:]  # assuming shape [..., H, W]

    pad_H = (unit - H % unit) % unit
    pad_W = (unit - W % unit) % unit

    pad_top = 0
    pad_bottom = pad_H
    pad_left = 0
    pad_right = pad_W

    img_padded = np.pad(
        img,
        pad_width=[(0, 0)] * (img.ndim - 2) + [(pad_top, pad_bottom), (pad_left, pad_right)],
        mode='reflect'  # or 'constant' for zero padding
    )
    return img_padded