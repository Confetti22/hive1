import random
import numpy as np
import torch
from torch.utils.data import Dataset


from helper.image_reader import Ims_Image

# You also need to modify your generate_sphereshell__shifts function to accept a `dims` argument.
def generate_sphereshell__shifts(R, r=0, dims=3):
    """Generate integer shifts within a sphere shell (radius R, inner radius r)."""
    shifts = []
    ranges = [range(-R, R+1)] * dims
    for shift in np.array(np.meshgrid(*ranges)).T.reshape(-1, dims):
        norm = np.linalg.norm(shift)
        if r < norm <= R:
            shifts.append(shift)
    return np.array(shifts)




class Contrastive_dataset_3d_one_stage(Dataset):
    """
    sample roi in ims file; return positive roi pairs and negative roi pairs for contrastive_learning
    """
    def __init__ (self,ims_path,d_near,num_pairs,n_view = 2,verbose = False):

        self.ims_vol =Ims_Image(ims_path,channel=0) 
        level = 0
        D,H,W= self.ims_vol.rois[level][3:]
        d_near = int(d_near) 

        margin = 10 
        # Generate random (x, y, z) locations within the given range
        lz, hz = d_near + margin +int(D//4),  int(D*3/4) - d_near - margin
        ly, hy = d_near + margin +int(H//4),  int(H*3/4) - d_near - margin
        lx, hx = d_near + margin +int(W//4),  int(W//2) - d_near - margin

        self.loc_lst = np.stack([
            np.random.randint(lz, hz, size=num_pairs),
            np.random.randint(ly, hy, size=num_pairs),
            np.random.randint(lx, hx, size=num_pairs)
        ], axis=1)

        self.sample_num = num_pairs
        self.all_near_shifts = generate_sphereshell__shifts(R= d_near,r= 24 ,dims=3)
        self.n_view =n_view


    def __len__(self):
        return self.sample_num
    
    def __getitem__(self,idx):

        z, y, x = self.loc_lst[idx].T    # Unpack coordinates
        roi = self.ims_vol.from_roi(coords=(z,y,x,64,64,64))
        roi = roi.astype(np.float32)
        roi=torch.from_numpy(roi)
        roi=torch.unsqueeze(roi,0)
        # for each call, the positive pair is resampled within the near_shifts range, 
        # maybe fix the positive pair will be better for stable training
        pair_locs = [self.positve_pair_loc_generate([z,y,x]) for _ in range(self.n_view -1)]
        pair_roi = [self.get_roi_given_loc(pair_loc) for pair_loc in pair_locs]
        res = [roi]
        for pair in pair_roi:
            res.append(pair)

        #res: x1,neigb1(x1),neigb2(x1),..,neigbN(x1)
        return res

    def get_roi_given_loc(self,loc):
        z, y, x = loc   # Unpack coordinates
        roi = self.ims_vol.from_roi(coords=(z,y,x,64,64,64))
        roi = roi.astype(np.float32)
        roi=torch.from_numpy(roi)
        roi=torch.unsqueeze(roi,0)
        return roi 

    def positve_pair_loc_generate(self,loc):
        shift = random.choice(self.all_near_shifts)
        return loc+shift


class Contrastive_dataset_3d_fix_paris(Dataset):
    """
    fix the positive pair at the init

    """
    def __init__(self, feats_map, d_near, num_pairs, n_view=2, verbose=False):
        D, H, W, C = feats_map.shape
        self.feats_map = feats_map
        d_near = int(d_near)
        margin = 10

        # Generate random (z, y, x) locations within the safe volume
        lz, hz = d_near + margin, D - d_near - margin
        ly, hy = d_near + margin, H - d_near - margin
        lx, hx = d_near + margin, W - d_near - margin

        self.loc_lst = np.stack([
            np.random.randint(lz, hz, size=num_pairs),
            np.random.randint(ly, hy, size=num_pairs),
            np.random.randint(lx, hx, size=num_pairs)
        ], axis=1)

        self.sample_num = num_pairs
        self.n_view = n_view
        self.all_near_shifts = generate_sphereshell__shifts(R=d_near, r=0, dims=3)

        # Fix positive pairs at initialization
        self.fixed_positive_pairs = [
            [self.positve_pair_loc_generate(self.loc_lst[i]) for _ in range(n_view - 1)]
            for i in range(self.sample_num)
        ]

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        z, y, x = self.loc_lst[idx]
        feat = self.feats_map[z, y, x, :]  # Shape: (C)
        feat = torch.from_numpy(feat)

        # Use fixed positive pairs
        pair_locs = self.fixed_positive_pairs[idx]
        pair_feats = [torch.from_numpy(self.get_feats_given_loc(loc, self.feats_map)) for loc in pair_locs]

        # res: x1, neigb1(x1), neigb2(x1), ..., neigbN(x1)
        return [feat] + pair_feats

    def get_feats_given_loc(self, loc, feat_map):
        z, y, x = loc
        return feat_map[z, y, x, :]

    def positve_pair_loc_generate(self, loc):
        shift = random.choice(self.all_near_shifts)
        loc = np.array(loc)
        new_loc = loc + shift
        return np.clip(new_loc, [0, 0, 0], np.array(self.feats_map.shape[:3]) - 1)  # Ensure valid bounds

class Contrastive_dataset_3d(Dataset):
    """
    Supports both 4-D (D, H, W, C) and 3-D (H, W, C) feature maps.

    Optional region-of-interest limits:
        lz, ly, lx – inclusive lower bounds
        hz, hy, hx – exclusive  upper bounds

    If any bound is None it is computed from d_near + margin.
    """
    def __init__(
        self,
        feats_map,
        d_near: int,
        num_pairs: int,
        n_view: int = 2,
        *,
        verbose: bool = False,
        margin: int = 10,
        lz: int | None = None,
        ly: int | None = None,
        lx: int | None = None,
        hz: int | None = None,
        hy: int | None = None,
        hx: int | None = None,
    ):
        self.feats_map = feats_map
        self.dims = feats_map.ndim - 1  # 3-D (volumetric) or 2-D (single slice)
        self.verbose = verbose
        self.n_view = n_view
        d_near = int(d_near)

        if self.dims == 3:
            D, H, W, C = feats_map.shape
            # --------- derive bounds (fallback to auto-computed) ----------
            lz = lz if lz is not None else d_near + margin 
            ly = ly if ly is not None else d_near + margin
            lx = lx if lx is not None else d_near + margin
            hz = hz if hz is not None else D - d_near - margin 
            hy = hy if hy is not None else H - d_near - margin
            hx = hx if hx is not None else W - d_near - margin

            if not (0 <= lz < hz <= D and 0 <= ly < hy <= H and 0 <= lx < hx <= W):
                raise ValueError("Sampling bounds are out of volume range.")

            self.loc_lst = np.stack([
                np.random.randint(lz, hz, size=num_pairs),
                np.random.randint(ly, hy, size=num_pairs),
                np.random.randint(lx, hx, size=num_pairs)
            ], axis=1)

        elif self.dims == 2:
            H, W, C = feats_map.shape
            ly = ly if ly is not None else d_near + margin
            lx = lx if lx is not None else d_near + margin
            hy = hy if hy is not None else H - d_near - margin
            hx = hx if hx is not None else W - d_near - margin

            if not (0 <= ly < hy <= H and 0 <= lx < hx <= W):
                raise ValueError("Sampling bounds are out of image range.")

            self.loc_lst = np.stack([
                np.random.randint(ly, hy, size=num_pairs),
                np.random.randint(lx, hx, size=num_pairs)
            ], axis=1)

        else:
            raise ValueError("Feature map must be 4-D (D, H, W, C) or 3-D (H, W, C).")

        self.sample_num = num_pairs
        self.all_near_shifts = generate_sphereshell__shifts(R=d_near, r=0, dims=self.dims)

    # ------------------------------------------------------------------ required
    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        loc = self.loc_lst[idx]
        feat = torch.from_numpy(self._get_feats_given_loc(loc)).float()
        pair_locs = [self._positive_pair_loc_generate(loc) for _ in range(self.n_view - 1)]
        pair_feats = [torch.from_numpy(self._get_feats_given_loc(pl)).float() for pl in pair_locs]
        return [feat] + pair_feats

    # ------------------------------------------------------------------ helpers
    def _get_feats_given_loc(self, loc):
        if self.dims == 3:
            z, y, x = loc
            return self.feats_map[z, y, x, :]
        else:  # dims == 2
            y, x = loc
            return self.feats_map[y, x, :]

    def _positive_pair_loc_generate(self, loc):
        shift = random.choice(self.all_near_shifts)
        return loc + shift

class Contrastive_dataset_multiple_2d(Dataset):
    def __init__ (self,feats_map,d_near,n_view = 2,verbose = False):

        N,C,H,W = feats_map.shape
        self.feats_maps = feats_map
        self.feats_map_shape = (C,H,W)
        self.data_length = N
        self.margin = 10

        d_near = int(d_near//(2**3*0.5))

        self.all_near_shifts = generate_sphereshell__shifts(R= d_near,r= 0, dims=2)
        self.n_view =n_view


    def __len__(self):
        return self.data_length
    
    def __getitem__(self,idx):

        C,H,W = self.feats_map_shape
        margin = self.margin
        ith_feat_map = self.feats_maps[idx]  # Shape: (C)
        y = random.randint(margin, H - margin - 1)
        x = random.randint(margin, W - margin - 1)
        sampled_feat = ith_feat_map[:,y,x]

        sampled_feat = torch.from_numpy(sampled_feat)
        # for each call, the positive pair is resampled within the near_shifts range, 
        # maybe fix the positive pair will be better for stable training
        pair_locs = [self.positve_pair_loc_generate([y,x]) for _ in range(self.n_view -1)]
        pair_feats = [ith_feat_map[:,pair_loc[0],pair_loc[1]] for pair_loc in pair_locs]
        res = [sampled_feat]
        for pair in pair_feats:
            res.append(torch.from_numpy(pair).float())

        #res: x1,neigb1(x1),neigb2(x1),..,neigbN(x1)
        return res

    def positve_pair_loc_generate(self,loc):
        shift = random.choice(self.all_near_shifts)
        return loc+shift
