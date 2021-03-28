# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy
import matplotlib.pyplot as plt
from numba import jit

MAX_ITERATIONS = 100
K4 = {'x': 3840, 'y': 2160}
K1 = {'x': 1920, 'y': 1080}

# ZOOM_POINT = {-0.670207263127854;0.45806578196347025;0.00048828125;}
# ZOOM_POINT = {'re': -0.5616438356164384, 'im': 0.6432648401826484, 'zm': 2}  #;100
ZOOM_POINT = {'re': 0, 'im': 0, 'zm': 1}


def render_image(columns, rows, image_name):
    zoom_factor = ZOOM_POINT['zm']
    zoom_dir = {'re': ZOOM_POINT['re'], 'im': ZOOM_POINT['im']}
    result = numpy.zeros([rows, columns])
    # Calc linspace bounds from zoom
    re_lower = -2  # -2 + zoom_dir['re']*zoom_factor
    re_upper = 1  # 1 + zoom_dir['re']*zoom_factor
    im_lower = -1  # -1 + zoom_dir['im']*zoom_factor
    im_upper = 1  # 1 + zoom_dir['im']*zoom_factor
    re_space = numpy.linspace(re_lower, re_upper, num=rows)
    im_space = numpy.linspace(im_lower, im_upper, num=columns)
    for row_index, re in enumerate(re_space):
        for column_index, im in enumerate(im_space):
            result[row_index, column_index] = mandelbrot_iteration(re, im, MAX_ITERATIONS)

    fig = plt.figure(figsize=(columns / 100, rows / 100), frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')

    plt.imshow(result.T, cmap='gist_stern', interpolation='bilinear', extent=[-2, 1, -1, 1])
    fig.savefig(f'{image_name}.png', transparent=True)
    plt.show()


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
    render_image(K1['x'], K1['y'], "mandelbrot")
