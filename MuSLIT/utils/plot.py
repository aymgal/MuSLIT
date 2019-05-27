__author__ = 'aymgal'


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def nice_colorbar(mappable, position='right', pad=0.1, size='5%', **kwargs):
    kwargs.update({'position': position, 'pad': pad, 'size': size})
    ax = mappable.axes
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(**kwargs)
    return plt.colorbar(mappable, cax=cax)


def plot_rgb_bands(color_image_np):
    band_r = color_image_np[:,:,0]
    band_g = color_image_np[:,:,1]
    band_b = color_image_np[:,:,2]
    vmax = max([band_r.max(), band_g.max(), band_b.max()])
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    ax = axes[0]
    im = ax.imshow(band_r, origin='lower', cmap='Reds_r', vmax=vmax)
    #fig.colorbar(im, ax=ax)
    ax = axes[1]
    ax.imshow(band_g, origin='lower', cmap='Greens_r', vmax=vmax)
    ax = axes[2]
    ax.imshow(band_b, origin='lower', cmap='Blues_r', vmax=vmax)
    ax = axes[3]
    ax.imshow(color_image_np, origin='lower')
    plt.show()

