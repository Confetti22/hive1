
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as v2
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np
import torch

class TraverseDataset3d(Dataset):
    def __init__(self, img, stride, win_size:int, net_input_shape:int, num_channel=1):
        """
        Handles 3D image volumes. If num_channel=3, repeats the image to 3 channels
        to adapt to networks pretrained on natural images.
        """
        self.img = img
        print(f"init traverseDataset with img of shape {img.shape},stride = {stride}, win_size = {win_size}")
        self.stride = stride
        self.win_size = win_size
        self.net_input_shape = net_input_shape
        # Expand to 3 channels if required
        # self.num_channel = num_channel
        # if img.shape[-1] != 3 and num_channel == 3:
        #     self.img = np.repeat(img[..., np.newaxis], 3, axis=-1)

        # Compute patches
        self.patches = self._generate_patches()

    def _generate_patches(self):
        """Generate 3D patches, with padding only if stride == 1."""
        if self.stride == 1:
            # Determine padding size
            half_size = self.win_size // 2
            if self.win_size % 2 == 0:
                pad_front = pad_top = pad_left = half_size - 1
                pad_back = pad_bottom = pad_right = half_size
            else:
                pad_front = pad_top = pad_left = pad_back = pad_bottom = pad_right = half_size

            # Apply padding
            padded_img = np.pad(
                self.img,
                pad_width=(
                    (pad_front, pad_back),  # Z-axis
                    (pad_top, pad_bottom),  # Y-axis
                    (pad_left, pad_right),  # X-axis
                    (0, 0),  # Channels
                ),
                mode="constant",
                constant_values=0,
            )
        else:
            # No padding
            padded_img = self.img

        # Extract patches
        patches = []
        for z in range(0, padded_img.shape[0] - self.win_size + 1, self.stride):
            for y in range(0, padded_img.shape[1] - self.win_size + 1, self.stride):
                for x in range(0, padded_img.shape[2] - self.win_size + 1, self.stride):
                    patch = padded_img[
                        z:z + self.win_size,
                        y:y + self.win_size,
                        x:x + self.win_size,
                    ]
                    patches.append(patch)
        
        

        self.sample_shape = np.array([ int(item//self.stride) +1 for item in [z,y,x]])
        print(f"sample shape = {self.sample_shape}")

        return patches
    def _get_sample_shape(self):
        return self.sample_shape

    def __len__(self):
        return len(self.patches)
    

    def __getitem__(self, idx):
        #TODO: synchronize the preprocess method used in training, right now, did not apply any norm or clip
        # Preprocess and resize the patch
        patch = self.patches[idx]
        # preprocess = v2.Compose([
        #     v2.Resize(size=self.net_input_shape),
        # ])
        patch = torch.tensor(patch,dtype=torch.float32)
        # patch = preprocess(patch)
        patch = torch.unsqueeze(patch,0)
        return patch
def get_feature_list(device,encoder,test_loader,extract_layer_name,save_path=None,apply_softmax=False)->np.ndarray:
    """
    encoder inference on a single input 2d-image
    
    input(numpy)--> test_dataset
    collect feats during inference
    return the feats as shape of N*n_dim

    """
    print(f"device is {device}")


    # a dict to store the activations
    activation = {}
    def getActivation(name):
        # the hook signature
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    #register the forward hook at layer"layer_name"
    # hook1 = getattr(encoder,extract_layer_name).register_forward_hook(getActivation(extract_layer_name))

    feats_list=[]
    for i, imgs in enumerate(tqdm(test_loader,desc="extracting features")):
        outs=encoder(imgs.to(device))
        if apply_softmax:
            outs = F.softmax(outs,dim=1)
            feats_list.append(outs.cpu().detach().numpy().reshape(outs.shape[0],-1))
        else:
            feats_list.append(activation[extract_layer_name].cpu().detach().numpy().reshape(outs.shape[0],-1))
    
    #detach the hook
    # hook1.remove()

    feats_array = np.concatenate([ arr for arr in feats_list], axis=0)
    print(f"fests_arry shape {feats_array.shape}")

    if save_path :
        with open(save_path, 'wb') as file:
            pickle.dump(feats_array, file)
    
    return feats_array


def contour_plt(img, comment='contour plot', save_path=None,writer=None, step=0):
    levels = [0.5]
    colors = ['white']
    levels = levels[::-1]
    colors = colors[::-1]
    
    if len(img.shape) ==3:
        half_z = int(img.shape[0]//2)
        img = img[half_z,:,:]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(img, interpolation='bilinear', origin='lower', cmap='viridis')
    cbar = fig.colorbar(im, ax=ax)
    CS = ax.contour(img, levels=levels, linewidths=0.8, colors=colors)
    ax.clabel(CS, inline=True, fontsize=10)
    ax.set_title(comment)

       # Save the plot to the specified path
    if save_path is not None:
        plt.savefig(save_path, format='png', dpi=300)
        print(f"Plot saved at {save_path}")

    # Log the plot to TensorBoard
    if writer:
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        # Convert the image to a tensor
        image = plt.imread(BytesIO(image), format='png')
        writer.add_image(comment, image, global_step=step, dataformats='HWC')
    else:
        plt.show()
    
    plt.close(fig)  # Close the figure to free memory
