from os import stat
import numpy as np
import math


L2 = np.linalg.norm  # defaults to L2


class Rotation:
    @staticmethod
    def _cross(u):
        # cross product matrix
        # https://stackoverflow.com/questions/66707295/
        return np.cross(u, -np.identity(3))

    def Rho(self, theta, u):
        # Rotation of theta degrees about u vector
        # https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
        return (
        + math.cos(theta) * np.identity(3)
        + math.sin(theta) * self._cross(u)
        + (1 - math.cos(theta)) * np.outer(u, u)
        ) if not np.isnan(u).any() else np.identity(3)

    @staticmethod
    def normal(u, v):
        # vector normal to u and v
        # returns nan vector if u and v are (close) to parallel or too short
        U = np.cross(u, v)
        L = L2(U)
        return U / L2(U) if L >= 1e-8 else np.full_like(U, fill_value=np.nan)


class Curve(Rotation):
    def __init__(self, x0=np.zeros(3), B=np.identity(3)):
        self.B = B
        self.x0 = x0

        self.x = [x0]

        self.Bs = [B]
        self.m = [x0]

    @staticmethod
    def proj_xy(theta, B):
        # vector CCW around xy plane of basis B
        return math.cos(theta)*B[0] + math.sin(theta)*B[1]

    @staticmethod
    def z(B):
        # Z component of basis matrix
        return B[2]

    def rot(self, u, v, n):
        # (CCW?) angle between vector u and v in [0, 360)
        # https://math.stackexchange.com/questions/878785/how-to-find-an-angle-in-range0-360-between-2-vectors
        dot = u @ v
        det = n @ np.cross(u, v)
        return np.arctan2(det, dot)

    def feed(self, sections):
        for length, segments, bend_angle, bend_axis in sections:
            bend_angle = math.radians(bend_angle)
            bend_axis = math.radians(bend_axis)

            l = length/segments  # segment length
            a = bend_angle/(segments+1)
            v = self.proj_xy(bend_axis, self.B)  # bend axis vector
            r = self.Rho(a, v) if not math.isnan(bend_axis) else np.identity(3)

            # azimuth is relative to y axis, not x axis
            azimuth = bend_axis - math.tau/4 if not np.isnan(bend_axis) else 0
            declination = bend_angle
            twist = -azimuth

            B = self.B.copy()
            B = B @ self.Rho(azimuth, B[2])
            B = B @ self.Rho(declination, B[1])
            B = B @ self.Rho(twist, B[2])
            # K == B

            # print(*map(math.degrees, (azimuth, declination, twist)))

            # apply the rotation in steps
            self.B = self.B @ r
            for _segment in range(segments):
                self.x.append(self.x[-1] + self.z(self.B) * l)
                self.B = self.B @ r

            # Rotation is perpendicular to Z-directions
            # so we first azimute around Z to align Y with the normal vector
            # then decline about the now perpendicular Y to align the Z vectors
            # then twist around Z to align the Y vectors
            # In case the Z vectors are already parallel, we only need to twist

            I = np.identity(3)
            n = self.normal(self.B[2], I[2])
            if np.isnan(n).any():
                azimuth = np.nan
                declination = np.nan
                twist = self.rot(I[1], self.B[1], I[2])
            else:
                azimuth = self.rot(I[1], n, I[2])
                declination = self.rot(self.B[2], I[2], n)
                twist = self.rot(n, self.B[1], self.B[2])

            print(*map(math.degrees, (azimuth, declination, twist)))

            self.m.append(self.x[-1])
            self.Bs.append(self.B)
        return self

class Curve2(Curve):
    # Certain parametrizations start the first bend with a half-angle
    def feed(self, sections):
        for length, segments, bend_angle, bend_axis in sections:
            bend_angle = math.radians(bend_angle)
            bend_axis = math.radians(bend_axis)

            l = length/segments
            a = bend_angle/segments/2
            v = self.proj_xy(bend_axis, self.B)
            r = self.Rho(a, v) if not math.isnan(bend_axis) else np.identity(3)
            # r = Rho(a, v) if bend_angle != 0.0 else np.identity(3)

            # B = B @ r
            for _segment in range(segments):
                self.B = self.B @ r
                self.x.append(self.x[-1] + self.z(self.B) * l)
                self.B = self.B @ r
            self.m.append(self.x[-1])
            self.Bs.append(self.B)
        return self


class Section(Rotation):
    def __init__(self):
        self.sections = []

    @staticmethod
    def A(x0, x1, x2):
        # Area between the three points
        # https://en.wikipedia.org/wiki/Cross_product
        return L2(np.cross(x1-x0, x2-x0)) / 2

    def R(self, x0, x1, x2):
        # Inverse of curvature defined by three points (radius)
        # https://en.wikipedia.org/wiki/Menger_curvature
        a = 4*self.A(x0, x1, x2)
        return np.inf if a == 0 else math.prod(map(L2, (x0-x1, x1-x2, x2-x0)))/a

    def curvature(self, x):
        # three-point curvature defined along a curve x
        # endpoints are nan, since they only have one neighbor
        x0, x1, *x = x
        yield math.nan
        while x:
            x2, *x = x
            yield 1 / self.R(x0, x1, x2)
            x0, x1 = x1, x2
        yield math.nan

    @staticmethod
    def split(curvature):
        # index of curve splits
        # defined by change of curvature
        # assumes all curves are circular
        # (straight lines are also circles)
        cp = np.nan
        f = False
        for i, e in enumerate(curvature):
            if f:
                cp = e
                f = False

            if abs(e - cp) >= 1e-8 or math.isnan(e):
                yield i
                f = True

    @staticmethod
    def xy_angle(p, B):
        # projected angle in xy-plane
        x = p @ B[0]
        y = p @ B[1]

        return (math.atan2(y, x) + math.tau/4) % math.tau\
            if L2(x+y) > 1e-6 else np.nan

    @staticmethod
    def arc_length(c, R):
        # arc length given chord and radius
        # https://en.wikipedia.org/wiki/Circular_segment
        return 2 * math.asin(np.clip(c / 2 / R, -1, 1))

    @staticmethod
    def phi(v1, v2):
        # angle between two vectors (cosine formula)
        return math.acos(np.clip(np.dot(v1, v2) / L2(v1) / L2(v2), -1, 1))

    def section(self, x, i):
        i0, *i = i

        B = np.identity(3)
        while i:
            i1, *i = i

            s = i1-i0  # segments

            l = sum(L2(x0-x1) for x0, x1 in zip(x[i0:i1], x[i0+1:]))  # length

            x0, x1, x2 = x[i0], x[i0+1], x[i1]

            c = L2(x2 - x0)  # chord length
            r1 = self.R(x0, x1, x2)  # radius of curvature
            bend_angle = self.arc_length(c, r1)  # * (s+1)/s
            # ^ s segments => s+1 bends          # ^ non-half-angle factor

            u = self.normal(x1-x0, x2-x0)  # normal on bend plane
            R = self.Rho(-bend_angle, u)  # rotation matrix

            B = B @ R

            bend_axis = self.xy_angle(x2-x0, B)

            self.sections.append(
                [s, l, np.degrees(bend_angle), np.degrees(bend_axis)])

            # new loop
            i0 = i1


    def feed(self, x):
        curvature = self.curvature(x)
        indexes = self.split(curvature)
        self.section(x, indexes)
        return self


sections = [
    (3, 3, 0, math.nan),
    (2, 10, 30, 90),
    (3, 10, -90, 0),
    (3, 3, 0, math.nan),
    (5, 10, 90, -30),
    # (10, 2, 90, 90),
    # (20, 8, 180, 180),
    # (10, 3, 0, math.nan),
    # (8, 4, 90, 40),
]

curve = Curve().feed(sections)
curve2 = Curve2().feed(sections)
# print(curve.Bs)

section = Section().feed(curve2.x)
# print(section.sections)



import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore", module="mpl_toolkits")

def plot_basis(ax, B, origin=np.zeros(3), axes="xyz", colors="k"):
    """Plots axis arrows"""
    for a, e in zip(axes, B):
        ax.quiver(*origin, *e, linewidths=0.8, colors=colors)
        ax.text(*(origin+e), s=a)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d", proj_type="ortho")

ax.plot(*zip(*curve.x))
# ax.plot(*zip(*curve2.x))

for x, B in zip(curve.m, curve.Bs):
    plot_basis(ax, B, x)

ax.set_xlim(-15, 15); ax.set_ylim(-15, 15); ax.set_zlim(-15, 15)
ax.set_box_aspect([1, 1, 1])
ax.azim = ax.elev = 0

plt.show()
