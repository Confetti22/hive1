#%%
import os
os.environ["NAPARI_ASYNC"] = "1"

import napari
import dask.array as da
from pathlib import Path
from dask_image.imread import imread
import re

# Set the target image directory

# Detect available channels by filename patterns (*C1.tif, *C2.tif, etc.)
def get_available_channels(directory: Path):
    pattern = re.compile(r".*C(\d).tif$")
    channels = set()
    for file in directory.rglob("*.tif"):
        match = pattern.match(file.name)
        if match:
            channels.add(f"C{match.group(1)}")
    return sorted(channels, key=lambda x: int(x[1]))  # C1, C2, C3, ...

# Load a stack of lazily-read Dask arrays for a given channel
def load_channel_stack(dir,channel: str):

    stack = imread(f"{dir}/*{channel}.tif")
    return stack



# Build stacks for all channels
if __name__ == '__main__':

    # directory = Path("/home/confetti/mnt/data/VISoR_Reconstruction/SIAT_SIAT/BiGuoqiang/Macaque_Brain/RM009_2/RM009_all_/ROIImage/4.0")
    directory = Path("/home/confetti/mnt/data/VISoR_Reconstruction/SIAT_SIAT/BiGuoqiang/Mouse_Brain/20210131_ZSS_USTC_THY1-YFP_1779_1/Reconstruction_1.0/Reconstruction/BrainImage/1.0")
    available_channels = get_available_channels(directory)
    print(f"Found channels: {available_channels}")

    channel_stacks = {}
    for channel in available_channels:
        print(f"Loading stack for {channel}...")
        stack = load_channel_stack(dir=directory,channel=channel)
        channel_stacks[channel] = stack

    # Launch Napari viewer and show all channel stacks
    viewer = napari.Viewer()
    for channel, stack in channel_stacks.items():
        viewer.add_image(stack, name=channel, contrast_limits=[0, 4000], multiscale=False)

    cs = list(viewer.dims.current_step)
    cs[0] = 12000
    viewer.dims.current_step = tuple(cs)
    napari.run()