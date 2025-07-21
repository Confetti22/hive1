from pathlib import Path
import numpy as np

try:
    import tifffile
except ImportError:
    tifffile = None

from skimage.transform import resize  # part of scikit-image

def upscale_mask_dir(
    in_dir: str,
    out_dir: str,
    patterns=(".tif", ".tiff", ".png"),
    scale=2,
    suffix="",
    overwrite=False,
    verbose=True,
):
    """
    Upscale each 2-D mask image in `in_dir` to (scale*H, scale*W) via nearest neighbor
    and save to `out_dir`.

    Parameters
    ----------
    in_dir : str
        Input directory containing mask images.
    out_dir : str
        Output directory (created if missing).
    patterns : tuple[str]
        Filename extensions to include.
    scale : int
        Linear scale factor (assumes isotropic x/y).
    suffix : str
        Optional suffix inserted before file extension in output filename.
    overwrite : bool
        If False, skip files already existing at destination.
    verbose : bool
        Print progress information.
    """
    in_path  = Path(in_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    files = [p for p in sorted(in_path.iterdir())
             if p.suffix.lower() in patterns and p.is_file()]

    if verbose:
        print(f"Found {len(files)} files.")

    for i, f in enumerate(files, 1):
        # Read
        if tifffile and f.suffix.lower() in (".tif", ".tiff"):
            arr = tifffile.imread(f)
        else:
            # Fallback via imageio or Pillow
            from PIL import Image
            arr = np.array(Image.open(f))

        if arr.ndim != 2:
            if verbose:
                print(f"[{i}/{len(files)}] Skip (not 2-D): {f.name}")
            continue

        H, W = arr.shape
        newH, newW = H * scale, W * scale

        # Nearest neighbor upscaling preserving labels
        up_arr = resize(
            arr,
            (newH, newW),
            order=0,              # nearest
            anti_aliasing=False,
            preserve_range=True
        ).astype(arr.dtype)

        # Compose output filename
        out_name = f.stem + suffix + f.suffix
        out_file = out_path / out_name

        if out_file.exists() and not overwrite:
            if verbose:
                print(f"[{i}/{len(files)}] Exists, skip: {out_file.name}")
            continue

        # Write
        if tifffile and out_file.suffix.lower() in (".tif", ".tiff"):
            tifffile.imwrite(out_file, up_arr)
        else:
            from PIL import Image
            Image.fromarray(up_arr).save(out_file)

        if verbose:
            print(f"[{i}/{len(files)}] {f.name} -> {out_file.name}  ({H}x{W} -> {newH}x{newW})")

    if verbose:
        print("Done.")

# ---- Example usage ----
if __name__ == "__main__":
    upscale_mask_dir(
        in_dir="/home/confetti/data/rm009/rm009_roi/single_layer/merged_bnds/before_smooth",
        out_dir="/home/confetti/data/rm009/rm009_roi/single_layer/merged_bnds/upscaled_before_smooth",
        scale=2,
        suffix="_2x",        # e.g. original: mask1.tif -> mask1_2x.tif
        overwrite=False,
        verbose=True
    )