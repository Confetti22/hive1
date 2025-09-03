import os
os.environ["NAPARI_ASYNC"] = "1"

import napari
import dask.array as da
from pathlib import Path
from dask_image.imread import imread
from tifffile import imread as tif_read
import re

def is_rgb_stack(directory: Path):
    tifs = sorted(directory.rglob("*.tif"))
    if not tifs:
        return False
    sample = tif_read(tifs[0])
    return sample.ndim == 3 and sample.shape[-1] == 3

def get_available_channels(directory: Path):
    pattern = re.compile(r".*C(\d).tif$")
    channels = set()
    for file in directory.rglob("*.tif"):
        match = pattern.match(file.name)
        if match:
            channels.add(f"C{match.group(1)}")
    return sorted(channels, key=lambda x: int(x[1]))

def load_channel_stack(dir, channel: str):
    stack = imread(f"{dir}/*{channel}.tif")
    return stack

def load_rgb_stack(directory: Path):
    files = sorted(directory.rglob("*.tif"))
    rgb_stack = da.stack([da.from_array(tif_read(str(f)), chunks='auto') for f in files], axis=0)
    return rgb_stack

if __name__ == '__main__':
    # directory = Path("/home/confetti/mnt/data/VISoR_Reconstruction/SIAT_SIAT/BiGuoqiang/Macaque_Brain/RM009_2/RM009_all_/ROIImage/4.0")
    # directory = Path("/home/confetti/data/rm009/rm009_roi/4")
    # directory = Path("/home/confetti/mnt/data/VISoR_Reconstruction/SIAT_SIAT/BiGuoqiang/Mouse_Brain/20210131_ZSS_USTC_THY1-YFP_1779_1/Reconstruction_1.0/Reconstruction/BrainImage/1.0")
    directory = Path("/home/confetti/data/dk/MD585/downsampled")

    if is_rgb_stack(directory):
        print("Detected RGB TIFF stack.")
        stack = load_rgb_stack(directory)
        viewer = napari.Viewer()
        viewer.add_image(stack, name="RGB", contrast_limits=[0, 4000], multiscale=False, rgb=True)
    else:
        print("Detected multi-channel grayscale TIFFs.")
        available_channels = get_available_channels(directory)
        print(f"Found channels: {available_channels}")

        channel_stacks = {}
        for channel in available_channels:
            print(f"Loading stack for {channel}...")
            stack = load_channel_stack(dir=directory, channel=channel)
            channel_stacks[channel] = stack

        viewer = napari.Viewer()
        for channel, stack in channel_stacks.items():
            viewer.add_image(stack, name=channel, contrast_limits=[0, 4000], multiscale=False)

    napari.run()