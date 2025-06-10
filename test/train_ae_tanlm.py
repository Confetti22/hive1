#%%
import tifffile
with tifffile.TiffFile('/home/confetti/data/tanlm/Z00026_488.tif') as tif:
    print(len(tif.pages))  # number of resolution levels
    for i, page in enumerate(tif.pages):
        print(f"Level {i}: shape = {page.shape}")