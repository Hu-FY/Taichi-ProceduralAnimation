import taichi as ti
import math

ti.init(arch = ti.gpu, debug = True)
res_x = 512
res_y = 512
canvas = ti.Vector.field(3, dtype = ti.f32, shape = (res_x, res_y))

@ti.kernel
def circle(radius:ti.f32, blur:ti.f32, tiles:ti.i32):
    clear()
    for i, j in canvas:
        size = res_x/tiles
        i_ = i % size
        j_ = j % size
        pos = ti.Vector([i_, j_])
        center = ti.Vector([res_x/2, res_y/2])
        r = (pos - center).norm()
        c = interpolate(1, 1-blur, r/radius)

        canvas[i, j] += c*ti.Vector([0, 1, 1])

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

gui = ti.GUI("Circle", res=(res_x, res_y))
iterations = 0
while gui.running:
    iterations += 1
    r = ti.sin(iterations/40)
    if r < 0: r = -r
    circle(r*(res_x*1.5), 1, 8)
    #test_inter()
    gui.set_image(canvas)
    gui.show()