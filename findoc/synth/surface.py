"""Curved documents as developable surfaces, and rendering them with exact ground truth (Phase 5).

Paper bends but does not stretch. Mathematically, a bent sheet is a **developable surface**: it
can be unrolled onto a plane with no distortion, so every length measured *on* the paper is
preserved. The simplest developable surfaces, and the ones that describe a curled, folded or
wavy cheque lying on a desk, are **generalised cylinders**: pick a direction ``a`` in the sheet;
every line of the sheet perpendicular to ``a`` stays straight, and the sheet bends only along
``a``, following a planar profile curve.

The profile is described by its **curvature** k(u) along the arc length u (distance measured on
the paper from the sheet's centre along ``a``). Integrating the curvature gives the tangent angle,
and integrating the tangent gives the curve:

    phi(u) = int_0^u k(s) ds,   x(u) = int_0^u cos(phi),   z(u) = -int_0^u sin(phi)

Because the curve is parametrised by arc length, the mapping from paper to 3D is an exact
isometry. That gives exact ground truth for Phase 5: the true "flattened" document is simply the
original texture.

* ``curl``: one or both ends lift with radius R beyond a distance from the centre (receipts and
  cheques straight out of a wallet);
* ``fold``: a crease of angle alpha with a small rounding radius (a cheque folded in half);
* ``wave``: gentle waviness (paper that got damp).

World coordinates follow :mod:`findoc.synth.camera`: the desk is the plane Z = 0, the camera is
at negative Z looking towards +Z, so "up, towards the camera" is negative Z.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from findoc.synth.camera import project


@dataclass
class Bend:
    kind: str = "flat"  # flat | curl | curl2 | fold | wave
    angle_deg: float = 0.0  # bend direction in the sheet (0 = along the width)
    radius: float = 400.0  # curl radius (texture pixels)
    start: float = 0.2  # curl/fold position as a fraction of the half-length from the centre
    fold_deg: float = 30.0  # fold angle
    fold_radius: float = 20.0  # rounding of the crease
    amplitude: float = 0.0004  # wave curvature amplitude (1/px)
    wavelength: float = 600.0

    def curvature(self, u: np.ndarray, half: float) -> np.ndarray:
        k = np.zeros_like(u)
        u0 = self.start * half
        if self.kind == "curl":  # one end lifts
            k[u > u0] = 1.0 / self.radius
        elif self.kind == "curl2":  # both ends lift
            k[u > u0] = 1.0 / self.radius
            k[u < -u0] = -1.0 / self.radius
        elif self.kind == "fold":
            length = np.deg2rad(self.fold_deg) * self.fold_radius
            k[(u > u0) & (u < u0 + length)] = 1.0 / self.fold_radius
        elif self.kind == "wave":
            k = self.amplitude * np.cos(2 * np.pi * u / self.wavelength)
        return k


@dataclass
class DocSurface:
    """A document of size (width, height) texture pixels bent by ``bend``, resting on the desk."""

    width: int
    height: int
    bend: Bend = field(default_factory=Bend)
    _profile: tuple[np.ndarray, np.ndarray, np.ndarray] | None = field(default=None, repr=False)

    def _curve(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._profile is None:
            a = np.deg2rad(self.bend.angle_deg)
            half = 0.5 * (abs(self.width * np.cos(a)) + abs(self.height * np.sin(a))) + 2
            u = np.linspace(-half, half, 4001)
            du = u[1] - u[0]
            k = self.bend.curvature(u, half)
            mid = len(u) // 2
            phi = np.zeros_like(u)
            phi[mid + 1 :] = np.cumsum(k[mid + 1 :]) * du
            phi[:mid] = -np.cumsum(k[:mid][::-1])[::-1] * du
            x = np.zeros_like(u)
            z = np.zeros_like(u)
            x[mid + 1 :] = np.cumsum(np.cos(phi[mid + 1 :])) * du
            z[mid + 1 :] = -np.cumsum(np.sin(phi[mid + 1 :])) * du
            x[:mid] = -np.cumsum(np.cos(phi[:mid][::-1]))[::-1] * du
            z[:mid] = np.cumsum(np.sin(phi[:mid][::-1]))[::-1] * du
            self._profile = (u, x, z)
        return self._profile

    def points(self, st: np.ndarray) -> np.ndarray:
        """Texture coordinates (N, 2) in pixels, origin top-left -> world points (N, 3)."""
        u_tab, x_tab, z_tab = self._curve()
        a = np.deg2rad(self.bend.angle_deg)
        da, dp = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
        p = st - np.array([self.width / 2, self.height / 2])
        u, v = p @ da, p @ dp
        x = np.interp(u, u_tab, x_tab)
        z = np.interp(u, u_tab, z_tab)
        xy = x[:, None] * da + v[:, None] * dp
        return np.column_stack([xy, z])

    def lift(self) -> float:
        """Height of the highest point above the desk (texture pixels)."""
        g = self.grid(40, 20)[0]
        z = self.points(g)[:, 2]
        return float(z.max() - z.min())

    def grid(self, nx: int = 64, ny: int = 32) -> tuple[np.ndarray, np.ndarray]:
        """Texture-space vertices (nx*ny, 2) and triangles (M, 3) of a regular mesh."""
        xs = np.linspace(0, self.width - 1, nx)
        ys = np.linspace(0, self.height - 1, ny)
        gx, gy = np.meshgrid(xs, ys)
        st = np.column_stack([gx.ravel(), gy.ravel()])
        tris = []
        for j in range(ny - 1):
            for i in range(nx - 1):
                a = j * nx + i
                tris += [(a, a + 1, a + nx), (a + 1, a + nx + 1, a + nx)]
        return st, np.array(tris, np.int32)

    def mesh(self, nx: int = 64, ny: int = 32) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(texture vertices, world vertices resting on the desk, triangles)."""
        st, tris = self.grid(nx, ny)
        world = self.points(st)
        world[:, 2] -= world[:, 2].max()  # lowest point touches the desk (Z = 0)
        return st, world, tris


def _triangle_shading(world: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Lambertian shading per triangle from a light above and slightly to the side."""
    e1 = world[tris[:, 1]] - world[tris[:, 0]]
    e2 = world[tris[:, 2]] - world[tris[:, 0]]
    n = np.cross(e1, e2)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    n *= np.sign(-n[:, 2:3] + 1e-12)  # point towards the camera side (negative Z)
    light = np.array([0.35, -0.3, -1.0])
    light /= np.linalg.norm(light)
    return 0.72 + 0.28 * np.clip(n @ light, 0, 1)


def render_surface(
    texture: np.ndarray,
    st: np.ndarray,
    world: np.ndarray,
    tris: np.ndarray,
    K: np.ndarray,  # noqa: N803
    pose,  # noqa: ANN001 - findoc.synth.camera.Pose
    background: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw the textured mesh over ``background`` (same size). Returns (image, document mask).

    Rasterisation by triangle IDs: triangles are painted far-to-near (painter's algorithm) into an
    integer ID image; every covered pixel then maps back to texture coordinates through its
    triangle's affine map, and the texture is sampled once with ``cv2.remap``.
    """
    h, w = background.shape[:2]
    img_pts = project(K, pose, world).astype(np.float32)
    depth = (world @ pose.R.T + pose.t)[:, 2]
    order = np.argsort(-depth[tris].mean(axis=1))  # far first
    ids = np.full((h, w), -1, np.int32)
    for t in order:
        cv2.fillConvexPoly(ids, img_pts[tris[t]].round().astype(np.int32), int(t))
    covered = ids >= 0
    if not covered.any():
        return background.copy(), covered
    # per-triangle affine map image -> texture, solved for all triangles at once:
    # [x y 1] @ A^T = [s t] for the three vertices
    m = np.concatenate([img_pts[tris].astype(np.float64), np.ones((len(tris), 3, 1))], axis=2)
    det = np.linalg.det(m)
    ok = np.abs(det) > 1e-6  # skip degenerate (edge-on) triangles
    aff = np.zeros((len(tris), 2, 3), np.float64)
    aff[ok] = np.transpose(np.linalg.solve(m[ok], st[tris[ok]].astype(np.float64)), (0, 2, 1))
    ys, xs = np.nonzero(covered)
    tid = ids[ys, xs]
    a = aff[tid]
    tx = a[:, 0, 0] * xs + a[:, 0, 1] * ys + a[:, 0, 2]
    ty = a[:, 1, 0] * xs + a[:, 1, 1] * ys + a[:, 1, 2]
    map_x = np.full((h, w), -1, np.float32)
    map_y = np.full((h, w), -1, np.float32)
    map_x[ys, xs], map_y[ys, xs] = tx, ty
    doc = cv2.remap(texture, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    shade = np.ones((h, w), np.float32)
    shade[ys, xs] = _triangle_shading(world, tris)[tid]
    out = background.astype(np.float32)
    m = covered[..., None]
    out = np.where(m, doc.astype(np.float32) * shade[..., None], out)
    return np.clip(out, 0, 255).astype(np.uint8), covered


BEND_KINDS = ("flat", "wave", "curl", "curl2", "fold")


def random_bend(kind: str, rng: np.random.Generator) -> Bend:
    """A random bend of the given kind, in the ranges used by the Phase 5 evaluation: gentle
    waves, one or both ends curled with radius 0.6-1.5 page widths, or a 15-40 degree fold."""
    angle = float(rng.choice([0.0, rng.uniform(-12, 12)]))
    if kind == "wave":
        return Bend(
            "wave", angle, amplitude=rng.uniform(0.001, 0.002), wavelength=rng.uniform(550, 800)
        )
    if kind == "curl":
        return Bend("curl", angle, radius=rng.uniform(600, 1100), start=rng.uniform(0.2, 0.5))
    if kind == "curl2":
        return Bend("curl2", angle, radius=rng.uniform(900, 1500), start=rng.uniform(0.3, 0.6))
    if kind == "fold":
        return Bend(
            "fold",
            angle,
            fold_deg=rng.uniform(15, 40),
            start=rng.uniform(-0.2, 0.2),
            fold_radius=25,
        )
    if kind != "flat":
        raise ValueError(f"unknown bend {kind!r}; expected one of {BEND_KINDS}")
    return Bend("flat")
