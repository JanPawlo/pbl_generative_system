"""
decal_projection.py

Implements 3D-aware decal application using the two-part texture mapping
approach described in Bier & Sloan (1986), "Two-Part Texture Mappings".

Mapping strategy: centroid/sphere
──────────────────────────────────────────────────────────────────────────────
The previous ISN/box version surrounded the mesh with a bounding box and used
six discrete face projections, producing hard seams at box edges where the
arbitration switched between faces.

centroid/sphere replaces the box intermediate surface with a bounding sphere.
The O⁻¹ mapping fires a ray from the mesh centroid through each surface point
and finds where it hits the sphere, giving continuous (θ, φ) spherical
coordinates.  Those are converted to equirectangular decal UV with no seams.

Trade-off (as noted in the paper): the S mapping from the flat decal image
onto the sphere has inherent stretch near the poles, but for decals covering
most of the object this is typically far less noticeable than the hard face
boundaries of the box approach.

Public API (unchanged):
  parse_obj(obj_path)  ->  ObjMesh
  apply_decal_to_model(texture_path, obj_path, decal_path, biome)  ->  None
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Linear-algebra helpers
# ---------------------------------------------------------------------------

Vec3 = np.ndarray  # shape (3,)


def _normalize(v: Vec3) -> Vec3:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("Cannot normalise a zero-length vector.")
    return v / n


def _vec3(x: float, y: float, z: float) -> Vec3:
    return np.array([x, y, z], dtype=np.float64)


# ---------------------------------------------------------------------------
# OBJ parser
# ---------------------------------------------------------------------------

@dataclass
class ObjMesh:
    """Minimal OBJ representation with per-triangle geometric normals."""
    positions:    np.ndarray   # (V, 3)  float64
    uvs:          np.ndarray   # (T, 2)  float64
    faces:        np.ndarray   # (F, 6)  int32  [vi0,vi1,vi2, ti0,ti1,ti2]
    face_normals: np.ndarray   # (F, 3)  float64  geometric normal per triangle


def parse_obj(obj_path: Path) -> ObjMesh:
    """
    Parse a Wavefront OBJ file and return triangulated geometry with UVs.
    Faces with more than 3 vertices are fan-triangulated.
    """
    positions: List[List[float]] = []
    uvs:       List[List[float]] = []
    faces:     List[List[int]]   = []

    with open(obj_path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            tag = parts[0]

            if tag == "v":
                positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif tag == "vt":
                uvs.append([float(parts[1]), float(parts[2])])
            elif tag == "f":
                verts = []
                for token in parts[1:]:
                    sub = token.split("/")
                    vi = int(sub[0]) - 1
                    ti = int(sub[1]) - 1 if len(sub) > 1 and sub[1] else -1
                    verts.append((vi, ti))
                for k in range(1, len(verts) - 1):
                    v0, t0 = verts[0]
                    v1, t1 = verts[k]
                    v2, t2 = verts[k + 1]
                    faces.append([v0, v1, v2, t0, t1, t2])

    if not positions:
        raise ValueError(f"OBJ file has no vertex data: {obj_path}")
    if not uvs:
        raise ValueError(
            f"OBJ file has no UV coordinates: {obj_path}. "
            "3D decal projection requires a UV-mapped mesh."
        )
    if not faces:
        raise ValueError(f"OBJ file has no face data: {obj_path}")

    pos_arr   = np.array(positions, dtype=np.float64)
    uvs_arr   = np.array(uvs,       dtype=np.float64)
    faces_arr = np.array(faces,     dtype=np.int32)

    # Geometric normals
    p0 = pos_arr[faces_arr[:, 0]]
    p1 = pos_arr[faces_arr[:, 1]]
    p2 = pos_arr[faces_arr[:, 2]]
    raw_n = np.cross(p1 - p0, p2 - p0)
    lengths = np.linalg.norm(raw_n, axis=1, keepdims=True)
    lengths = np.where(lengths < 1e-12, 1.0, lengths)
    face_normals = raw_n / lengths

    return ObjMesh(
        positions=pos_arr,
        uvs=uvs_arr,
        faces=faces_arr,
        face_normals=face_normals,
    )


# ---------------------------------------------------------------------------
# centroid/sphere mapping
# ---------------------------------------------------------------------------

@dataclass
class SphereDecalSetup:
    """Bounding sphere that wraps the mesh for centroid/sphere projection."""
    centroid: np.ndarray   # (3,)  centre of the sphere
    radius:   float


def _rotate_vectors_y(vectors: np.ndarray, angle: float) -> np.ndarray:
    """Rotate (N, 3) unit vectors around the Y axis by *angle* radians."""
    if angle == 0.0:
        return vectors
    c = np.cos(angle)
    s = np.sin(angle)
    x = vectors[:, 0]
    y = vectors[:, 1]
    z = vectors[:, 2]
    return np.stack([c * x + s * z, y, -s * x + c * z], axis=-1)


def _build_sphere_setup(mesh: ObjMesh) -> SphereDecalSetup:
    """
    Fit a bounding sphere to *mesh* using its centroid and the furthest vertex.
    """
    centroid = mesh.positions.mean(axis=0)
    dists    = np.linalg.norm(mesh.positions - centroid, axis=1)
    radius   = float(dists.max())
    radius   = max(radius, 1e-4)
    return SphereDecalSetup(centroid=centroid, radius=radius)


def _project_to_sphere_uv(
    pts:        np.ndarray,        # (N, 3)  3D surface points
    setup:      SphereDecalSetup,
    rotation_y: float = 0.0,
) -> np.ndarray:                    # (N, 2)  decal UV in [0, 1]
    """
    centroid/sphere O⁻¹ mapping (Bier & Sloan §"Centroid/sphere").

    For each surface point, fire a ray from the centroid through the point and
    find where it intersects the bounding sphere.  Convert the intersection's
    spherical coordinates (θ, φ) to equirectangular UV.

    O⁻¹: [xo, yo, zo]  →  r * [xo, yo, zo] / ||[xo, yo, zo]||

    UV mapping:
      u = (atan2(z, x) / (2π)) + 0.5   (longitude → [0, 1], seam at ±X)
      v = acos(y / r) / π               (latitude  → [0, 1], 0=north, 1=south)

    The only discontinuity is a single vertical seam at the ±X boundary
    (where atan2 wraps), much less visible than the six box-face seams.
    """
    d = pts - setup.centroid              # (N, 3)  direction from centroid

    # Normalise to get the unit direction, then scale to sphere surface
    norms = np.linalg.norm(d, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    d_unit = d / norms                    # (N, 3)
    if rotation_y != 0.0:
        d_unit = _rotate_vectors_y(d_unit, rotation_y)

    xs, ys, zs = d_unit[:, 0], d_unit[:, 1], d_unit[:, 2]

    # Spherical → equirectangular UV
    u = np.arctan2(zs, xs) / (2.0 * np.pi) + 0.5   # longitude [0, 1]
    v = np.arccos(np.clip(ys, -1.0, 1.0)) / np.pi   # latitude  [0, 1]

    return np.stack([u, v], axis=-1)                  # (N, 2)


# ---------------------------------------------------------------------------
# UV-atlas rasteriser
# ---------------------------------------------------------------------------

def _build_uv_to_3d_map(
    mesh: ObjMesh, tex_w: int, tex_h: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rasterise the OBJ UV atlas into per-texel 3D data.

    Returns
    -------
    pos3d   : (tex_h, tex_w, 3)  float32  barycentric-interpolated 3D positions
    normals : (tex_h, tex_w, 3)  float32  geometric face normal per texel
    Both arrays have NaN for texels that fall outside all UV triangles.
    """
    pos3d   = np.full((tex_h, tex_w, 3), np.nan, dtype=np.float32)
    normals = np.full((tex_h, tex_w, 3), np.nan, dtype=np.float32)

    positions    = mesh.positions.astype(np.float32)
    uvs          = mesh.uvs.astype(np.float32)
    faces        = mesh.faces
    face_normals = mesh.face_normals.astype(np.float32)

    px_u = (np.arange(tex_w, dtype=np.float32) + 0.5) / tex_w
    px_v = (np.arange(tex_h, dtype=np.float32) + 0.5) / tex_h

    for fi, face in enumerate(faces):
        vi0, vi1, vi2, ti0, ti1, ti2 = face
        if ti0 < 0 or ti1 < 0 or ti2 < 0:
            continue

        uv0 = uvs[ti0];  uv1 = uvs[ti1];  uv2 = uvs[ti2]
        p0  = positions[vi0]
        p1  = positions[vi1]
        p2  = positions[vi2]
        fn  = face_normals[fi]

        min_u = max(0,         int(min(uv0[0], uv1[0], uv2[0]) * tex_w) - 1)
        max_u = min(tex_w - 1, int(max(uv0[0], uv1[0], uv2[0]) * tex_w) + 1)
        min_v = max(0,         int((1.0 - max(uv0[1], uv1[1], uv2[1])) * tex_h) - 1)
        max_v = min(tex_h - 1, int((1.0 - min(uv0[1], uv1[1], uv2[1])) * tex_h) + 1)

        if min_u > max_u or min_v > max_v:
            continue

        gu = px_u[min_u: max_u + 1]
        gv = 1.0 - px_v[min_v: max_v + 1]   # flip V: OBJ bottom-up, image top-down
        GU, GV = np.meshgrid(gu, gv)

        dU = uv1 - uv0;  dV = uv2 - uv0
        det = float(dU[0] * dV[1] - dU[1] * dV[0])
        if abs(det) < 1e-10:
            continue

        qU = GU - uv0[0];  qV = GV - uv0[1]
        lam1 = (qU * dV[1] - qV * dV[0]) / det
        lam2 = (qU * (-dU[1]) + qV * dU[0]) / det
        lam0 = 1.0 - lam1 - lam2

        inside = (lam0 >= 0) & (lam1 >= 0) & (lam2 >= 0)
        if not inside.any():
            continue

        interp = (
            lam0[..., np.newaxis] * p0 +
            lam1[..., np.newaxis] * p1 +
            lam2[..., np.newaxis] * p2
        )

        rows = np.arange(min_v, max_v + 1)
        cols = np.arange(min_u, max_u + 1)
        R, C = np.meshgrid(rows, cols, indexing="ij")

        pos3d  [R[inside], C[inside]] = interp[inside]
        normals[R[inside], C[inside]] = fn

    return pos3d, normals


# ---------------------------------------------------------------------------
# Bilinear sampler
# ---------------------------------------------------------------------------

def _bilinear_sample(
    arr: np.ndarray,   # (H, W, C)  float32
    xs:  np.ndarray,   # (N,)       float32  fractional pixel x
    ys:  np.ndarray,   # (N,)       float32  fractional pixel y
) -> np.ndarray:       # (N, C)
    H, W, _ = arr.shape
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    fx = (xs - x0).astype(np.float32)[:, np.newaxis]
    fy = (ys - y0).astype(np.float32)[:, np.newaxis]

    x0c = np.clip(x0,     0, W - 1);  x1c = np.clip(x0 + 1, 0, W - 1)
    y0c = np.clip(y0,     0, H - 1);  y1c = np.clip(y0 + 1, 0, H - 1)

    samples = (
        arr[y0c, x0c] * (1 - fx) * (1 - fy) +
        arr[y0c, x1c] *      fx  * (1 - fy) +
        arr[y1c, x0c] * (1 - fx) *      fy  +
        arr[y1c, x1c] *      fx  *      fy
    )
    oob = (x0 < 0) | (x0 >= W) | (y0 < 0) | (y0 >= H)
    samples[oob] = 0.0
    return samples


# ---------------------------------------------------------------------------
# Main compositing function
# ---------------------------------------------------------------------------

def apply_decal_3d(
    base_img:     Image.Image,
    decal_img:    Image.Image,
    mesh:         ObjMesh,
    biome:        dict,
    rotation_y:   float = 0.0,
) -> Image.Image:
    """
    Apply *decal_img* onto *base_img* using centroid/sphere mapping.

    Steps
    -----
    1. Rasterise the OBJ UV atlas to get per-texel 3D positions.
    2. Fit a bounding sphere to the mesh.
    3. For each texel, fire a ray from the sphere centroid through the surface
       point and convert the intersection to equirectangular (θ, φ) UV.
    4. Bilinear-sample the decal; alpha-composite over the base texture.
    """
    tex_w, tex_h = base_img.size
    dec_w, dec_h = decal_img.size

    # 1. UV atlas -> 3D positions (normals not needed for centroid/sphere)
    pos3d, _ = _build_uv_to_3d_map(mesh, tex_w, tex_h)

    valid_mask = ~np.isnan(pos3d[..., 0])
    valid_pts  = pos3d[valid_mask].astype(np.float64)

    if valid_pts.shape[0] == 0:
        return base_img.copy()

    # 2. Bounding sphere
    setup = _build_sphere_setup(mesh)

    # 3. Project onto sphere -> equirectangular decal UV
    decal_uv = _project_to_sphere_uv(valid_pts, setup, rotation_y=rotation_y)

    dec_px_x = (decal_uv[:, 0] * dec_w - 0.5).astype(np.float32)
    dec_px_y = (decal_uv[:, 1] * dec_h - 0.5).astype(np.float32)

    # 5. Sample decal and composite
    decal_arr = np.array(decal_img, dtype=np.float32)
    samples   = _bilinear_sample(decal_arr, dec_px_x, dec_px_y)

    result_arr = np.array(base_img, dtype=np.float32)
    rows_idx, cols_idx = np.where(valid_mask)
    base_px = result_arr[rows_idx, cols_idx]

    src_a   = samples[:, 3:4] / 255.0
    out_rgb = samples[:, :3] * src_a + base_px[:, :3] * (1.0 - src_a)
    out_a   = samples[:, 3:4] + base_px[:, 3:4] * (1.0 - src_a)

    result_arr[rows_idx, cols_idx] = np.concatenate([out_rgb, out_a], axis=1)
    return Image.fromarray(np.clip(result_arr, 0, 255).astype(np.uint8), mode="RGBA")


# ---------------------------------------------------------------------------
# Flat (tiled-UV) decal apply — simple alpha-composite with rotation
# ---------------------------------------------------------------------------

def apply_decal_flat(
    base_img:   Image.Image,
    decal_img:  Image.Image,
) -> Image.Image:
    """
    Apply *decal_img* onto *base_img* by scaling the decal to fill the texture
    and alpha-compositing the result. No rotation, no 3D projection.

    Suitable for tiled-UV objects where every face already shows the full
    texture and should receive the same decal stamp.
    """
    tex_w, tex_h = base_img.size
    scaled = decal_img.resize((tex_w, tex_h), Image.Resampling.LANCZOS)
    result = base_img.copy().convert("RGBA")
    result.alpha_composite(scaled.convert("RGBA"))
    return result


# ---------------------------------------------------------------------------
# Public wrapper
# ---------------------------------------------------------------------------

def read_decal_mode(mtl_path: Path) -> str:
    """
    Read the first  '# decal_mode <mode>'  comment from an MTL file.
    Returns 'sphere' if no such comment is found (safe default).

    To mark a model as flat/tiled, add this line anywhere in its .mtl file:
        # decal_mode flat
    To use the sphere projection explicitly (or by default), use:
        # decal_mode sphere
    """
    try:
        for raw in mtl_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if line.startswith("# decal_mode"):
                parts = line.split()
                if len(parts) >= 3:
                    return parts[2].lower()
    except OSError:
        pass
    return "sphere"


def apply_decal_to_model(
    texture_path: Path,
    obj_path:     Path,
    decal_path:   Path,
    biome:        dict,
    rotation_y:   float = 0.0,
    mode:         str   = "sphere",
) -> None:
    """
    Load the model texture, apply the decal, and save in place.

    mode='sphere'  →  centroid/sphere 3D projection (default, complex meshes)
    mode='flat'    →  direct alpha-composite with rotation (tiled-UV objects)

    The mode is normally determined by  read_decal_mode(mtl_path)  in the
    caller; pass it explicitly here so this function stays side-effect-free.
    """
    base_img  = Image.open(texture_path).convert("RGBA")
    decal_img = Image.open(decal_path).convert("RGBA")

    if mode == "flat":
        result = apply_decal_flat(base_img, decal_img)
    else:
        mesh   = parse_obj(obj_path)
        result = apply_decal_3d(base_img, decal_img, mesh, biome, rotation_y=rotation_y)

    result.save(texture_path)