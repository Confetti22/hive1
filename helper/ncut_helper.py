import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries
from skimage import graph
from skimage import color
from scipy.ndimage import zoom
from sklearn.decomposition import PCA

from skimage.graph import RAG
import math

from confettii.plot_helper import grid_plot_list_imgs
def rag_mean_feature(image, labels, connectivity=2, mode='similarity', sigma=0.001):
    # Initialize RAG
    rag = RAG(labels, connectivity=connectivity)

    h, w, c = image.shape  # C is number of feature channels
    for n in rag:
        rag.nodes[n].update({
            'labels': [n],
            'pixel count': 0,
            'total feature': np.zeros((c,), dtype=np.float64),
        })

    # Accumulate feature sums and counts
    for index in np.ndindex(labels.shape):
        current = labels[index]
        rag.nodes[current]['pixel count'] += 1
        rag.nodes[current]['total feature'] += image[index]

    # Compute mean feature vector per region
    for n in rag:
        rag.nodes[n]['mean feature'] = (
            rag.nodes[n]['total feature'] / rag.nodes[n]['pixel count']
        )

    # Compute weights based on feature vector distance
    for x, y, d in rag.edges(data=True):
        diff = rag.nodes[x]['mean feature'] - rag.nodes[y]['mean feature']
        diff_norm = np.linalg.norm(diff)  # Euclidean distance
        if mode == 'similarity':
            d['weight'] = math.e ** (-(diff_norm**2) / sigma)
        elif mode == 'distance':
            d['weight'] = diff_norm
        else:
            raise ValueError(f"The mode '{mode}' is not recognised")

    return rag

def segment_and_plot_from_feats(
    feats_map,
    image,
    label = None,
    rag = None,
    slic_compactness=0.3,
    rag_weight_sigma=0.01,
    n_segments=100,
    slic_iters=30,
    ncut_thresh=0.001
):
    """
    Segments image using a feature map and visualizes PCA features, superpixels, RAG, and normalized cuts.

    Parameters:
    - feats_map: np.ndarray of shape (H, W, C) — Feature map
    - image: np.ndarray — Original image (H, W[, 3])
    - slic_compactness: float — Compactness for SLIC superpixels
    - rag_weight_sigma: float — Sigma for RAG edge weights
    - n_segments: int — Desired number of superpixels
    - slic_iters: int — Max iterations for SLIC
    - ncut_thresh: float — Threshold for normalized cut
    """
    
    # Match feature map resolution to input image
    zoom_factors = tuple(raw / feat for raw, feat in zip(image.shape[:2], feats_map.shape[:2]))
    feats_map_rescaled = zoom(feats_map, zoom=(*zoom_factors, 1), order=1)

    # Normalize features
    feats_map_rescaled = (feats_map_rescaled - feats_map_rescaled.min()) / (feats_map_rescaled.max() - feats_map_rescaled.min())
    
    # SLIC Superpixels
    if label is None: 
        label = slic(
            feats_map_rescaled,
            n_segments=n_segments,
            compactness=slic_compactness,
            max_num_iter=slic_iters,
            start_label=1,
            channel_axis=-1
        )

    # Build RAG and apply normalized cut
    if rag is None:
        rag = rag_mean_feature(feats_map_rescaled, label, mode='similarity', sigma=rag_weight_sigma)
    labels = graph.cut_normalized(label, rag, thresh=ncut_thresh)

    # PCA for visualization
    h, w, c = feats_map_rescaled.shape
    flat_feats = feats_map_rescaled.reshape(-1, c)
    pca = PCA(n_components=3)
    if c <= 3:
        rgb_vis = feats_map_rescaled
    else:
        rgb_vis = pca.fit_transform(flat_feats).reshape(h, w, 3)
        rgb_vis = (rgb_vis - rgb_vis.min()) / (rgb_vis.max() - rgb_vis.min())

    grid_plot_list_imgs(
        images=[image,rgb_vis],
        ncols=2,
        fig_size=6
    )


    # Visualization
    fig, ax = plt.subplots(ncols=4, sharex=True, sharey=True, figsize=(24, 24))

    ax[0].set_title('Image')
    ax[0].imshow(image, cmap='gray' if image.ndim == 2 else None)

    ax[1].set_title('Superpixels')
    ax[1].imshow(mark_boundaries(rgb_vis, label,mode='inner'))

    ax[2].set_title('RAG')
    lc = graph.show_rag(
        label,
        rag,
        rgb_vis,
        border_color='yellow',
        img_cmap='gray',
        edge_cmap='coolwarm',
        edge_width=1,
        ax=ax[2]
    )

    ax[3].set_title('Normalized Cut')
    ax[3].imshow(color.label2rgb(labels, rgb_vis, kind='avg'))

    for a in ax:
        a.axis('off')

    plt.tight_layout()
    plt.show()
