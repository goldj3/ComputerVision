import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button, RadioButtons

fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(left=0.25, bottom=0.25)
min0 = 0
max0 = 255

im = np.zeros((512,512, 3), np.uint8)
print(im.shape)
im1 = ax.imshow(im)


axcolor = 'lightgoldenrodyellow'
red_loc = fig.add_axes([0.25, 0.1, 0.65, 0.03])
green_loc  = fig.add_axes([0.25, 0.15, 0.65, 0.03])
blue_loc  = fig.add_axes([0.25, 0.20, 0.65, 0.03])


red_slider = Slider(red_loc, 'Red', 0, 255, valinit=0)
green_slider = Slider(green_loc, 'Green', 0, 255, valinit=0)
blue_slider = Slider(blue_loc, 'Blue', 0, 255, valinit=0)


def update(val):
    im1.set_data((1,1,1))
    fig.canvas.draw()

red_slider.on_changed(update)
green_slider.on_changed(update)
blue_slider.on_changed(update)


plt.show()