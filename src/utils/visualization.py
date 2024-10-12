import time
import warnings

import distinctipy
import matplotlib.pyplot as plt
import numpy as np
import visdom

warnings.filterwarnings("ignore")

plt.style.use("ggplot")
plt.rcParams["figure.facecolor"] = "#171717"
plt.rcParams["text.color"] = "#DDDDDD"


def display_image_channels(image, channels, title="Image Channels"):
    # print('x',image.shape)
    channel_names = channels
    fig, axes = plt.subplots(1, 4, figsize=(10, 5))
    for idx, ax in enumerate(axes.flatten()):
        channel_image = image[idx, :, :]  # Transpose the array to display the channel
        ax.imshow(channel_image, cmap="gray")
        ax.axis("off")
        ax.set_title(channel_names[idx])
    plt.suptitle(title, fontsize=20, y=0.93)
    plt.tight_layout()
    # plt.suptitle(title, fontsize=20, y=1.03)
    return fig


def display_mask_channels_as_rgb(
    mask, channel_names, colors, title="Mask Channels as RGB"
):
    n_channels = mask.shape[0]
    fig, axes = plt.subplots(1, n_channels, figsize=(10, 5))
    assert len(colors) == n_channels
    colors = [np.array(color, dtype="float32") for color in colors]
    # print(colors)
    for idx in range(n_channels):
        # convert from 2D grayscale image to 2D multichannel (RGB)
        rgb_mask = np.stack((mask[idx],) * 3, -1) * 255
        # print(rgb_mask.shape)
        rgb_mask_rows, rgb_mask_cols = np.where(rgb_mask[:, :, 1] == 255)
        rgb_mask[rgb_mask_rows, rgb_mask_cols, :] = colors[idx] * 255
        axes[idx].imshow(rgb_mask)
        axes[idx].axis("off")
        axes[idx].set_title(channel_names[idx])
    plt.suptitle(title, fontsize=20, y=0.93)
    plt.tight_layout()
    return fig


def normalize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min) / (_max - _min)
    return normalized_img


def plot_image_and_mask(image, mask, **kwargs):
    # 4d mask and image
    # print(image.shape)
    # print(mask.shape)
    depth = kwargs.get("depth")
    region_names = kwargs.get("subregions")
    channels = kwargs.get("im_channels")
    mask_region_colors = kwargs.get("mask_region_colors")
    vis = kwargs["vis"]
    title = kwargs["title"]

    if not depth:
        depth = image.shape[-1] // 2
    image = image[:, :, :, depth]
    mask = mask[:, :, :, depth]

    # print(image.shape)
    # print(mask.shape)

    excolors = [(0, 0, 0), (255, 255, 255)]
    if not mask_region_colors:
        mask_region_colors = distinctipy.get_colors(len(region_names), excolors)

    fig_imgs = display_image_channels(image, channels)
    fig_masks = display_mask_channels_as_rgb(
        mask, colors=mask_region_colors, title=title, channel_names=region_names
    )
    # plot_overlay_masks_on_image(image, mask)
    vis.matplot(
        fig_imgs, opts={"resizable": True, "height": 200, "width": 400}, win="img"
    )
    vis.matplot(
        fig_masks, opts={"resizable": True, "height": 200, "width": 400}, win="masks"
    )
    plt.close(fig_imgs)
    plt.close(fig_masks)
    # time.sleep(1)


def plot_mask(mask, **kwargs):
    # 4d mask
    depth = kwargs.get("depth")
    region_names = kwargs.get("subregions")
    mask_region_colors = kwargs.get("mask_region_colors")
    vis = kwargs["vis"]
    title = kwargs["title"]
    win = kwargs["win"]

    if not depth:
        depth = mask.shape[-1] // 2
    mask = mask[:, :, :, depth]

    fig_masks = display_mask_channels_as_rgb(
        mask, colors=mask_region_colors, title=title, channel_names=region_names
    )
    vis.matplot(
        fig_masks, opts={"resizable": True, "height": 200, "width": 400}, win=win
    )
    plt.close(fig_masks)
