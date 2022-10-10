# Shadertoy "Fractal Tiling", reference ==> https://www.shadertoy.com/view/Ml2GWy#

import taichi as ti
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import handy_shader_functions as hsf #import the handy shader functions from the parent folder

ti.init(arch=ti.cpu)

res_x = 768
res_y = 512
pixels = ti.Vector.field(3, ti.f32, shape=(res_x, res_y))


@ti.kernel
def render(t: ti.f32):
    for i_, j_ in pixels:
        color = ti.Vector([0.0, 0.0, 0.0])
        
        tile_size = 3

        offset = int(t*5) # make it move

        num_tiles = 8
        for k in range(6):
            size = res_x / num_tiles
            i = (i_ + offset) // size
            j = (j_ + offset) // size
            c = (0.5 * ti.sin(50 * float(i) + 38 * float(j) + 0.5*t + 10*float(i * j) + float(i**2) + float(j**2) + k) + 0.5)/2**(k+1)
            x = (i_ + offset) % size
            y = (j_ + offset) % size
            weight = 1-max(x/size, 1-x/size)**16-max(y/size, 1-y/size)**16
            center = ti.Vector([size/2, size/2])
            pos = ti.Vector([x, y])
            radius = (pos-center).norm()
            weight *= radius * 2 / size + 0.5
            c *= weight
            color += ti.Vector([c, c*0.8, c*0.8])
            num_tiles *= 2

        color = hsf.clamp(color, 0.0, 1.0)

        pixels[i_, j_] = color

gui = ti.GUI("Fractal Tiling", res=(res_x, res_y))
i = 0
"""
while gui.running:
    render(i*0.05)
    gui.set_image(pixels)
    gui.show()
    i = i+1
"""
video_manager = ti.tools.VideoManager(output_dir="./output", framerate=30, automatic_build=False)
for i in range(300):
    render(i*0.05)
    img = pixels.to_numpy()
    video_manager.write_frame(img)

video_manager.make_video(gif=True, mp4=True)