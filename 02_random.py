import taichi as ti
import math

ti.init(arch = ti.gpu)
res_x = 512
res_y = 512
canvas = ti.Vector.field(3, dtype = ti.f32, shape = (res_x, res_y))

@ti.kernel
def circle(radius: ti.f32, tiles: ti.i32, iterations: ti.i32):
    clear()
    size = res_x/tiles
    for t in ti.static(range(3)):
        center = ti.Vector([size/2, size/2])
        for i, j in canvas:
            a = iterations * 0.03
            r = (0.5 * ti.sin(a + float(i) * 5 / res_x + t) + 0.5)/(t+1)
            g = (0.5 * ti.sin(a + float(j) * 5 / res_y + 2 + t) + 0.5)/(t+1)
            b = (0.5 * ti.sin(a + float(i) * 5 / res_x + 4 + t) + 0.5)/(t+1)
            color = ti.Vector([r, g, b])
            i_ = i % size
            j_ = j % size
            pos = ti.Vector([i_, j_])
            r = (pos - center).norm()
            blur = ti.sin((0.01*iterations+i//size*5+j//size*3))
            blur -= ti.floor(blur)
            blur = (blur+0.001)*0.9
            c = interpolate(1, 1-blur, r/(radius/2**t))
            canvas[i, j] += c*color
        size *= 0.5

@ti.func
def interpolate(start, end, pos):
    r = (pos-start)/(end-start)
    r = min(max(r,0),1)
    return (-2*r**2)*(r-3/2)

@ti.func
def clear():
    for i, j in canvas:
        canvas[i, j] = ti.Vector([0, 0, 0])

@ti.kernel
def test_inter():
    for i, j in canvas:
        canvas[i, j] = ti.Vector([0,0,0])
    for x in range(res_x):
        y = interpolate(res_x/3, 2*res_x/3, x)
        canvas[x, int(y*(res_y-1))] = ti.Vector([1,1,1])

gui = ti.GUI("Circles", res=(res_x, res_y))
iterations = 0
tiles = 8
tilesize = res_x/tiles
"""
while gui.running:
    iterations += 1
    r = ti.sin(iterations/100)
    if r < 0: r = -r
    circle(r*tilesize/2, tiles, iterations)
    #test_inter()
    gui.set_image(canvas)
    gui.show()
"""
video_manager = ti.tools.VideoManager(output_dir="./output", framerate=24, automatic_build=False)
for i in range(315):
    iterations += 1
    r = ti.sin(iterations/100)
    if r < 0: r = -r
    circle(r*tilesize/2, tiles, iterations)
    img = canvas.to_numpy()
    video_manager.write_frame(img)

video_manager.make_video(gif=True, mp4=True)

