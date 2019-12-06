import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons


blank = np.zeros((512,512,3), np.uint8)
plt.imshow(blank)
axcolor = 'lightgoldenrodyellow'
axfreq = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
axamp = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)

t = np.arange(0.0, 1.0, 0.001)
a0 = 0
f0 = 0
delta_f = 5.0
red_slider = Slider(axfreq, 'Red', 0.1, 30.0, valinit=0, valstep=1)
green_slider = Slider(axfreq, 'Green', 0.1, 30.0, valinit=0, valstep=1)
blue_slider = Slider(axfreq, 'Blue', 0.1, 30.0, valinit=0, valstep=1)
plt.show()


fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
t = np.arange(0.0, 1.0, 0.001)
a0 = 0
f0 = 0
delta_f = 5.0
s = a0 * np.sin(2 * np.pi * f0 * t)
l, = plt.plot(t, s, lw=2)
ax.margins(x=0)





samp = Slider(axamp, 'Amp', 0.1, 10.0, valinit=a0)
sfreq = Slider(axfreq, 'Freq', 0.1, 30.0, valinit=f0, valstep=delta_f)


def update(val):
    amp = samp.val
    freq = sfreq.val
    l.set_ydata(amp*np.sin(2*np.pi*freq*t))
    fig.canvas.draw_idle()


sfreq.on_changed(update)
samp.on_changed(update)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


def reset(event):
    red_slider.reset()
    green_slider.reset()
    blue_slider.reset()
    sfreq.reset()
    samp.reset()
button.on_clicked(reset)

rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)


def colorfunc(label):
    l.set_color(label)
    fig.canvas.draw_idle()

radio.on_clicked(colorfunc)
plt.show()