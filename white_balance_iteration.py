import picamera
import picamera.array
import numpy as np
import operator

def cropND(img, bounding):
    start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]

with picamera.PiCamera() as camera:
    camera.resolution = (1280, 720)
    camera.awb_mode = 'off'
    # Start off with ridiculously low gains
    rg, bg = (0.5, 0.5)
    camera.awb_gains = (rg, bg)
    with picamera.array.PiRGBArray(camera, size=(1280, 720)) as output:
        # Allow 30 attempts to fix AWB
        for i in range(350):
            # Capture a tiny resized image in RGB format, and extract the
            # average R, G, and B values
            camera.capture(output, format='rgb', use_video_port=True)
            output.array = cropND(output.array, (200, 300))
            r, g, b = (np.mean(output.array[..., i]) for i in range(3))
            print('R:%5.2f, B:%5.2f = (%5.2f, %5.2f, %5.2f)' % (
                rg, bg, r, g, b))
            # Adjust R and B relative to G, but only if they're significantly
            # different (delta +/- 2)
            if abs(r - g) > 1:
                if r > g:
                    rg -= 0.01
                else:
                    rg += 0.01
            if abs(b - g) > 1:
                if b > g:
                    bg -= 0.01
                else:
                    bg += 0.01
            camera.awb_gains = (rg, bg)
            output.seek(0)
            output.truncate()
