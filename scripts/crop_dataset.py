# ----------------------------------
# scripts/crop_rois.py (Step 1: crop)
# ----------------------------------
# Reads:   paths.input_image, paths.level, paths.channel, crop.*
# Writes:  runs/<run_id>/data/<ae_train_dir>/*.tif and .../<ae_test_dir>/*.tif


import argparse
from pathlib import Path
import numpy as np
import tifffile as tif

# If you have your project helper available:
try:
    from helper.image_reader import Ims_Image
except Exception:
    Ims_Image = None


def shannon_entropy(img):
    # Lightweight replacement for skimage.measure.shannon_entropy
    vals, counts = np.unique(img, return_counts=True)
    p = counts.astype(np.float64) / counts.sum()
    return float(-(p * np.log2(p + 1e-12)).sum())


def entropy_filter(l_thres=1.4, h_thres=100.0):
    def _filter(img):
        ent = shannon_entropy(img)
        return (ent >= l_thres) and (ent <= h_thres)
    return _filter



def load_cfg(cfg_path):
    import yaml
    with open(cfg_path, 'r') as f:
        return yaml.safe_load(f)


def make_split_ranges(sample_range, axis: int, ratio: float, v: bool = True):
    sr = [list(r) for r in sample_range]
    lb, ub = sr[axis]
    cut = int(ratio * (ub - lb)) + lb
    sr_train = [tuple(r) for r in sr]
    sr_test = [tuple(r) for r in sr]
    sr_train[axis] = [lb, cut]
    sr_test[axis] = [cut, ub]
    if v:   
        print(f"split axis {axis} @ {cut}: train {sr_train[axis]}  test {sr_test[axis]}")             
    return sr_train, sr_test


#Todo: and support muliple file formats:1.ims(3d); 2.tif(3d);3.dir of tiffs(2d), 4.tiff(2d) 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    args = ap.parse_args()
    cfg = load_cfg(args.cfg)

    crop = cfg['crop']
    paths = cfg['paths']

    # Build run-scoped output dirs: runs/<run_id>/data/<ae_train_dir|ae_test_dir>
    run_id = cfg['run_id']
    root = Path(paths.get('output_root', 'runs')).expanduser().absolute() / run_id
    data_root = root / 'data'
    train_dir = data_root / paths['ae_train_dir']
    test_dir  = data_root / paths['ae_test_dir']
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    img_path = Path(paths['input_image'])
    level = int(paths.get('level', 0))
    channel = int(paths.get('channel', 0))

    roi_size = tuple(crop['rois_size'])
    amount = int(crop['amount'])
    test_amount = max(1, int(round(amount * (1 - float(crop['train_split'])))))
    train_amount = amount
    ent_th = float(crop['entropy_thres'])
    sample_range = crop.get('sample_range', None)
    split_ratio = float(crop['train_split'])
    split_axis = int(crop.get('split_axis', 0))  # 0,1,2 → which axis to split along
    if isinstance(sample_range, str) and sample_range.lower() == 'none':
        sample_range = None

    # load source volume
    if img_path.suffix.lower() == '.ims':
        if Ims_Image is None:
            raise SystemExit('[ERR] .ims input requires helper.image_reader.Ims_Image')
        ims = Ims_Image(str(img_path), channel=channel)
        shape = ims.info[level]['data_shape']

        def _get_random_roi_ims(sample_range):
            roi,indexs=ims.get_random_roi(filter=entropy_filter(l_thres=ent_th),
                                            roi_size=roi_size,level=level,skip_gap =False,
                                            sample_range=sample_range,margin=0)
            return roi
        
        # Default full range then split along chosen axis
        if sample_range is None:
            sample_range = [[0, shape[0]], [0, shape[1]], [0, shape[2]]]

        train_sample_range, test_sample_range = make_split_ranges(sample_range, split_axis, split_ratio)

        get_train_roi = lambda: _get_random_roi_ims(train_sample_range)
        get_test_roi  = lambda: _get_random_roi_ims(test_sample_range)
  


    else:
        vol = tif.imread(str(img_path))  # expects (Z,Y,X)
        shape = vol.shape

        filter_fn = entropy_filter(l_thres=ent_th)
        def _get_random_roi(sample_range):
            foreground_sample_flag = False 
            while not foreground_sample_flag:
                margin = 0
                sample_lb = [idx_range[0] for idx_range in sample_range]
                sample_rb = [ idx_range[1] for idx_range in sample_range]
                z_idx=np.random.randint(sample_lb[0] + margin ,sample_rb[0]-roi_size[0] - margin) 
                y_idx=np.random.randint(sample_lb[1] + margin ,sample_rb[1]-roi_size[1] - margin) 
                x_idx=np.random.randint(sample_lb[2] + margin ,sample_rb[2]-roi_size[2] - margin) 

                roi= vol[z_idx:z_idx+roi_size[0], y_idx:y_idx+roi_size[1], x_idx:x_idx + roi_size[2]]
                #filter check
                foreground_sample_flag=filter_fn(roi)

            return roi

        if sample_range is None:
            sample_range = [[0, shape[0]], [0, shape[1]], [0, shape[2]]]

        train_sample_range, test_sample_range = make_split_ranges(sample_range, split_axis, split_ratio)

        get_train_roi = lambda: _get_random_roi(train_sample_range)
        get_test_roi  = lambda: _get_random_roi(test_sample_range)




    # train/test writing

    train_cnt = 0
    test_cnt = 0

    # fill train first, then test
    while train_cnt < train_amount:
        roi = get_train_roi()
        if roi is None:
            raise SystemExit('[ERR] Could not find enough ROIs passing entropy filter; relax thresholds/sizes.')
        tif.imwrite(str(train_dir / f"{train_cnt:05d}.tif"), roi)
        train_cnt += 1
        if train_cnt % 200 == 0:
            print(f"[crop] saved train {train_cnt}/{train_amount}")

    while test_cnt < test_amount:
        roi = get_test_roi()
        if roi is None:
            raise SystemExit('[ERR] Could not find enough ROIs for TEST; relax thresholds/sizes.')
        tif.imwrite(str(test_dir / f"{test_cnt:05d}.tif"), roi)
        test_cnt += 1
        if test_cnt % 100 == 0:
            print(f"[crop] saved test {test_cnt}/{test_amount}")

    print(f"[OK] crop done → train:{train_cnt}  test:{test_cnt}")

if __name__ == '__main__':
    main()
