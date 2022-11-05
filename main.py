import os
import numpy
import matplotlib.pyplot as plt
from numba import jit
import cv2
import glob

MAX_ITERATIONS = 100
K4 = (3840, 2160)
K2w = (2560, 1920)
K1 = (1920, 1080)

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

# Some zoompoints
Z1 = numpy.array([-1.5, 0.3])


# Currently not aspect ratio aware. Only works well with ultra wide like K2w
def render_image(
    frame_size, zoom_point, zoom_factor=0.0, color_map="CMRmap_r", show_img=False
):
    width, height = frame_size
    # Define view port and zoom
    z = zoom_point  # zoom point
    # Scale viewport
    mm_z = mm + (z - mm) * zoom_factor
    mp_z = mp + (z - mp) * zoom_factor
    pp_z = pp + (z - pp) * zoom_factor
    pm_z = pm + (z - pm) * zoom_factor

    re_space = numpy.linspace(
        mm_z[0], pp_z[0], num=height
    )  # lower left and lower right real components
    im_space = numpy.linspace(
        mm_z[1], mp_z[1], num=width
    )  # lower left and upper left img components

    # Init result array
    result = numpy.zeros([height, width])
    # Enumerate over all points
    for row_index, re in enumerate(re_space):
        for column_index, im in enumerate(im_space):
            result[row_index, column_index] = mandelbrot_iteration(
                re, im, MAX_ITERATIONS
            )

    # Set up figure for rendering
    # width and height divided by constant to make them the size of a pixel
    fig = plt.figure(figsize=(width / 100, height / 100), frameon=False)
    # Add and disable axes
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    plt.imshow(
        numpy.flipud(result.T),
        cmap=color_map,
        interpolation="gaussian",
        aspect="auto",
        extent=[-2, 1, -1.5, 1.5],
    )

    if show_img:
        plt.show()

    return fig


@jit("u4(f4, f4, u4)")
def mandelbrot_iteration(re, im, max_iter):
    c = complex(re, im)  # Our input complex number
    z = 0.0j  # The iteration number

    for i in range(max_iter):
        z = z * z + c
        if (z.real * z.real + z.imag * z.imag) >= 4:
            return i
    return max_iter


# duration in seconds
def write_images(
    file_path: str,
    zoom_point,
    zoom_start,
    zoom_end,
    duration=10,
    fps=60,
    frame_size=K2w,
    image_format="png",
    debug=False,
):
    # Ensure dir exists
    d = os.path.dirname(file_path)
    if not os.path.exists(d):
        os.makedirs(d)

    # Create zoom stages via a linear space
    zoom_space = numpy.linspace(
        zoom_start, zoom_end, num=duration * fps
    )  # lower left and lower right real components

    for img_index, zoom_factor in enumerate(zoom_space):
        fig = render_image(frame_size, zoom_point, zoom_factor)
        fig.savefig(
            f"{file_path}({zoom_point[0]},{zoom_point[1]})-{zoom_start}_{zoom_end}-{duration}@{fps}"
            f"-{frame_size[0]}X{frame_size[1]}-#{img_index}.{image_format}",
            transparent=True,
        )
        plt.close(fig)
        if debug:
            print(
                f"Rendered image #{img_index+1:>3}/{len(zoom_space)} ({(img_index+1)/len(zoom_space)}%)"
            )


def write_video(
    file_path: str,
    zoom_point,
    zoom_start,
    zoom_end,
    duration=10,
    fps=60,
    frame_size=K2w,
    image_format="png",
    images_path=None,
    debug=False,
):
    if not images_path:
        images_path = file_path

    # Render all images
    write_images(
        images_path,
        zoom_point,
        zoom_start,
        zoom_end,
        duration,
        fps,
        frame_size,
        image_format,
        debug=debug,
    )

    print("Rendering video...")

    # Assemble video
    writer = cv2.VideoWriter(
        f"{file_path}mandelbrot.avi", cv2.VideoWriter_fourcc(*"DIVX"), fps, frame_size
    )
    image_list = glob.glob(f"{images_path}*.{image_format}")
    image_list.reverse()
    for filename in image_list:
        img = cv2.imread(filename)
        writer.write(img)
    writer.release()

    print("Done")


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    render_image(K1, Z1, show_img=True)
    # write_video('/tmp/mandelbrot/test2/', Z1, 0, 0.999, 1, 6, debug=True)
