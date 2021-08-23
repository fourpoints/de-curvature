import math
import numpy as np


L2 = np.linalg.norm  # defaults to L2

def A(x0, x1, x2):
    # https://en.wikipedia.org/wiki/Cross_product
    return L2(np.cross(x1-x0, x2-x0)) / 2


def n(x0, x1, x2):
    U = np.cross(x1-x0, x2-x0)
    L = L2(U)
    return U / L2(U) if L >= 1e-8 else np.full_like(U, fill_value=np.nan)


def R(x0, x1, x2):
    # https://en.wikipedia.org/wiki/Menger_curvature
    a = 4*A(x0, x1, x2)
    return math.prod(map(L2, (x0-x1, x1-x2, x2-x0))) / a if a != 0.0 else np.inf


def theta(c, R):
    # https://en.wikipedia.org/wiki/Circular_segment
    return 2 * math.asin(c / 2 / R)


def phi(v1, v2):
    return math.acos(np.clip(np.dot(v1, v2) / L2(v1) / L2(v2), -1, 1))


def xy_angle(p, B):
    # projected angle in xy-plane
    x = p @ B[0]
    y = p @ B[1]

    return (math.atan2(y, x) - math.tau/4) % math.tau\
        if L2(x+y) > 1e-6 else np.nan


def _x(u):
    # https://stackoverflow.com/questions/66707295/
    return np.cross(u, -np.identity(3))


def Rho(theta, u):
    # https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    return (
      + math.cos(theta) * np.identity(3)
      + math.sin(theta) * _x(u)
      + (1 - math.cos(theta)) * np.outer(u, u)
    ) if not np.isnan(u).any() else np.identity(3)


def u(theta, B):
    return math.cos(theta)*B[0] + math.sin(theta)*B[1]


def z(B):
    return B[2]


def o(v):
    r = np.random.randn(3)
    return v * r.dot(v) / np.linalg.norm(v)**2


def curvature(x):
    x0, x1, *x = x
    yield math.nan
    while x:
        x2, *x = x
        yield 1 / R(x0, x1, x2)
        x0, x1 = x1, x2
    yield math.nan


def arc_split(c):
    cp = np.nan
    f = False
    for i, e in enumerate(c):
        if f:
            cp = e
            f = False

        if abs(e - cp) >= 1e-8 or math.isnan(e):
            yield i
            f = True


def arc_sections(x, i):
    i0, *i = i

    B = np.identity(3)
    while i:
        i1, *i = i

        s = i1-i0  # segments

        l = sum(L2(x0-x1) for x0, x1 in zip(x[i0:i1], x[i0+1:]))  # length

        x0, x1, x2 = x[i0], x[i0+1], x[i1]

        c = L2(x2-x0)  # chord
        r1 = R(x0, x1, x2)  # radius of curvature
        a = theta(c, r1) * (s+1)/s # bend_angle
        # ^ bend angle is divided by segments + 1

        u = n(x0, x1, x2)  # normal on bend plane

        # align basis with curve
        z1 = (x2-x0) @ Rho(a/2, u)
        b = phi(z1, B[2])
        v = n(np.zeros(3), z1, B[2])
        v = v if not np.isnan(v).any() else B[0]  # use x-axis if 180 flip
        B = B @ Rho(b, v)


        p = xy_angle(x2-x0, B)  # bend_axis
        # B = B @ Rho(-a, u)  # alternate method; assumes initial basis is same


        # new loop
        i0 = i1

        print(f"{l:.2f}, {s:.2f}, {math.degrees(a):.2f}, {math.degrees(p):.2f}")



if __name__ == "__main__":
    B = np.identity(3)
    x0 = np.zeros(3)

    x = [x0]

    Bs = [B]
    m = [x0]

    sections = [
        (10, 5, 90, 20),
        (20, 8, 180, 180),
        (10, 3, 0, math.nan),
        (8, 4, 90, 40),
    ]

    for length, segments, bend_angle, bend_axis in sections:
        bend_angle = math.radians(bend_angle)
        bend_axis = math.radians(bend_axis)

        l = length/segments
        a = bend_angle/(segments+1)
        v = u(bend_axis, B)
        r = Rho(a, v) if bend_angle != 0.0 else np.identity(3)
        B = B @ r
        for _segment in range(segments):
            x.append(x[-1] + z(B) * l)
            B = B @ r
        m.append(x[-1])
        Bs.append(B)


    cs = list(curvature(x))
    indexes = list(arc_split(cs))
    arc_sections(x, indexes)
