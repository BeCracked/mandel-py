import numpy
import matplotlib.pyplot as plt
from numba import jit

MAX_ITERATIONS = 100
K4 = {'x': 3840, 'y': 2160}
K2w = {'x': 2560, 'y': 1920}
K1 = {'x': 1920, 'y': 1080}

# Base linear space values
b_re_lower = -2
b_re_upper = 1
b_im_lower = -1.5
b_im_upper = 1.5

# View port base coordinates
mm = numpy.array([b_re_lower, b_im_lower])  # lower left corner
mp = numpy.array([b_re_lower, b_im_upper])  # upper left corner
pp = numpy.array([b_re_upper, b_im_upper])  # upper right corner
pm = numpy.array([b_re_upper, b_im_lower])  # lower right corner


# Currently not aspect ratio aware. Only works well with ultra wide like K2w
def render_image(width=2560, height=1920, zoom_point=numpy.array([-1, 0.3]), zoom_factor=0., color_map='CMRmap_r'):
    # Define view port and zoom
    z = zoom_point  # zoom point
    # Scale viewport
    mm_z = mm + (z - mm) * zoom_factor
    mp_z = mp + (z - mp) * zoom_factor
    pp_z = pp + (z - pp) * zoom_factor
    pm_z = pm + (z - pm) * zoom_factor

    re_space = numpy.linspace(mm_z[0], pp_z[0], num=height)  # lower left and lower right real components
    im_space = numpy.linspace(mm_z[1], mp_z[1], num=width)  # lower left and upper left img components

    # Init result array
    result = numpy.zeros([height, width])
    # Enumerate over all points
    for row_index, re in enumerate(re_space):
        for column_index, im in enumerate(im_space):
            result[row_index, column_index] = mandelbrot_iteration(re, im, MAX_ITERATIONS)

    # Set up figure for rendering
    # width and height divided by constant to make them the size of a pixel
    fig = plt.figure(figsize=(width / 100, height / 100), frameon=False)
    # Add and disable axes
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')

    plt.imshow(numpy.flipud(result.T), cmap=color_map, interpolation='gaussian',
               aspect='auto', extent=[-2, 1, -1.5, 1.5])
    fig.savefig(f'mandelbrot.png', transparent=True)
    plt.show()

    return plt


@jit("u4(f4, f4, u4)")
def mandelbrot_iteration(re, im, max_iter):
    c = complex(re, im)  # Our input complex number
    z = 0.0j  # The iteration number

    for i in range(max_iter):
        z = z * z + c
        if (z.real * z.real + z.imag * z.imag) >= 4:
            return i
    return max_iter


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    img = render_image(K2w['x'], K2w['y'], numpy.array([-1, 0.3]), 0.99)
