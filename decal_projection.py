"""
decal_projection.py

Implements 3D-aware decal application using the two-part texture mapping
approach described in Bier & Sloan (1986), "Two-Part Texture Mappings".

Mapping strategy: ISN/box (Intermediate Surface Normal, box intermediate surface)
─────────────────────────────────────────────────────────────────────────────────
The previous version used a single slide-projector (plane) pass, which only
covered faces roughly perpendicular to the one chosen projection direction.
On a cube this meant only top/bottom faces received the decal.

ISN/box fixes this by surrounding the mesh with an axis-aligned bounding box
and using SIX projectors — one per box face.  For each texel we compute which
box face's inward normal is closest to the surface normal at that point
(the arbitration scheme from the paper), then sample the decal from that face's
projection.  This gives continuous, full-coverage results on any convex mesh
(and acceptable results on mildly non-convex ones).

Public API (unchanged from previous version):
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
# ISN/box setup
# ---------------------------------------------------------------------------

# Six outward-facing normals of an axis-aligned box, and a consistent up-hint
# per face so the decal artwork has a predictable orientation everywhere.
#
#   index  face    normal      up-hint
#     0    +X     [ 1, 0, 0]  [0, 1, 0]
#     1    -X     [-1, 0, 0]  [0, 1, 0]
#     2    +Y     [ 0, 1, 0]  [0, 0, 1]   (top)
#     3    -Y     [ 0,-1, 0]  [0, 0, 1]   (bottom)
#     4    +Z     [ 0, 0, 1]  [0, 1, 0]
#     5    -Z     [ 0, 0,-1]  [0, 1, 0]

_BOX_NORMALS: np.ndarray = np.array([
    [ 1,  0,  0],
    [-1,  0,  0],
    [ 0,  1,  0],
    [ 0, -1,  0],
    [ 0,  0,  1],
    [ 0,  0, -1],
], dtype=np.float64)

_BOX_UPS: np.ndarray = np.array([
    [0, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 0],
], dtype=np.float64)


@dataclass
class BoxDecalSetup:
    """Pre-computed local frames for all six faces of the bounding box."""
    centers:  np.ndarray   # (6, 3)
    x_axes:   np.ndarray   # (6, 3)
    y_axes:   np.ndarray   # (6, 3)
    half_ws:  np.ndarray   # (6,)   half-width  along x_axis
    half_hs:  np.ndarray   # (6,)   half-height along y_axis


def _build_box_setup(mesh: ObjMesh, padding: float = 0.0) -> BoxDecalSetup:
    """
    Build a BoxDecalSetup that snugly wraps *mesh*.

    padding expands the box on all sides so the projection stays well-defined
    right up to the mesh boundary.  The paper recommends using a box with side
    = sqrt(2) * sphere_radius for minimal data loss; here we use a proportional
    padding on the actual bounding box.
    """
    lo = mesh.positions.min(axis=0) - padding
    hi = mesh.positions.max(axis=0) + padding

    cx, cy, cz = (lo + hi) / 2.0
    hx, hy, hz = (hi - lo) / 2.0

    centers = np.array([
        [hi[0], cy,    cz   ],   # +X face
        [lo[0], cy,    cz   ],   # -X face
        [cx,    hi[1], cz   ],   # +Y face
        [cx,    lo[1], cz   ],   # -Y face
        [cx,    cy,    hi[2]],   # +Z face
        [cx,    cy,    lo[2]],   # -Z face
    ], dtype=np.float64)

    # Each face spans two axes; map those to half_w / half_h
    # ±X face: spans Y (width) × Z (height)
    # ±Y face: spans X (width) × Z (height)
    # ±Z face: spans X (width) × Y (height)
    half_ws = np.array([hy, hy, hx, hx, hx, hx], dtype=np.float64)
    half_hs = np.array([hz, hz, hz, hz, hy, hy], dtype=np.float64)

    x_axes = np.empty((6, 3), dtype=np.float64)
    y_axes = np.empty((6, 3), dtype=np.float64)
    for i in range(6):
        n  = _BOX_NORMALS[i]
        up = _BOX_UPS[i]
        x  = _normalize(np.cross(up, n))
        y  = _normalize(np.cross(n, x))
        x_axes[i] = x
        y_axes[i] = y

    return BoxDecalSetup(
        centers=centers,
        x_axes=x_axes,
        y_axes=y_axes,
        half_ws=half_ws,
        half_hs=half_hs,
    )


def _arbitrate_face(surface_normals: np.ndarray) -> np.ndarray:
    """
    For each surface point pick the box face whose outward normal best aligns
    with the surface normal (Bier & Sloan arbitration scheme).

    surface_normals : (N, 3)
    Returns         : (N,)  int  in [0, 5]
    """
    dots = surface_normals @ _BOX_NORMALS.T   # (N, 6)
    return np.argmax(dots, axis=1)            # (N,)


def _project_to_face_uv(
    pts:      np.ndarray,      # (N, 3)
    face_ids: np.ndarray,      # (N,)
    setup:    BoxDecalSetup,
) -> np.ndarray:               # (N, 2)  decal UV in [0, 1]
    """
    Project each 3D point onto its assigned box face and return UV coordinates.
    """
    uv = np.empty((len(pts), 2), dtype=np.float64)

    for fi in range(6):
        mask = face_ids == fi
        if not mask.any():
            continue
        d  = pts[mask] - setup.centers[fi]
        xa = d @ setup.x_axes[fi]
        ya = d @ setup.y_axes[fi]
        uv[mask, 0] = xa / (2.0 * setup.half_ws[fi]) + 0.5
        uv[mask, 1] = ya / (2.0 * setup.half_hs[fi]) + 0.5

    return uv


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
    base_img:  Image.Image,
    decal_img: Image.Image,
    mesh:      ObjMesh,
    biome:     dict,
) -> Image.Image:
    """
    Apply *decal_img* onto *base_img* using ISN/box mapping.

    Steps
    -----
    1. Rasterise the OBJ UV atlas to get per-texel 3D positions and normals.
    2. Build an axis-aligned bounding box around the mesh (with sqrt(2) padding).
    3. For each texel, pick the box face whose normal best matches the surface
       normal (arbitration).
    4. Project the texel's 3D point onto that face to get decal [xa, ya].
    5. Bilinear-sample the decal; alpha-composite over the base texture.
    """
    tex_w, tex_h = base_img.size
    dec_w, dec_h = decal_img.size

    # 1. UV atlas -> 3D positions + normals
    pos3d, surf_normals = _build_uv_to_3d_map(mesh, tex_w, tex_h)

    valid_mask = ~np.isnan(pos3d[..., 0])
    valid_pts  = pos3d[valid_mask].astype(np.float64)
    valid_nrm  = surf_normals[valid_mask].astype(np.float64)

    if valid_pts.shape[0] == 0:
        return base_img.copy()

    # 2. Bounding box — pad by (sqrt(2)/2 - 0.5) * max_extent so the box
    #    side is approximately sqrt(2) * half-extent, per the paper's advice.
    extents = mesh.positions.max(axis=0) - mesh.positions.min(axis=0)
    padding = float(np.max(extents)) * (np.sqrt(2.0) / 2.0 - 0.5)
    padding = max(padding, 1e-4)
    setup = _build_box_setup(mesh, padding=padding)

    # 3. Arbitrate: choose which box face each texel belongs to
    face_ids = _arbitrate_face(valid_nrm)

    # 4. Project onto box face -> decal UV
    decal_uv = _project_to_face_uv(valid_pts, face_ids, setup)

    dec_px_x = (decal_uv[:, 0] * dec_w - 0.5).astype(np.float32)
    dec_px_y = ((1.0 - decal_uv[:, 1]) * dec_h - 0.5).astype(np.float32)

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
# Public wrapper (API unchanged from previous version)
# ---------------------------------------------------------------------------

def apply_decal_to_model(
    texture_path: Path,
    obj_path:     Path,
    decal_path:   Path,
    biome:        dict,
) -> None:
    """
    Load the model texture, apply the decal via ISN/box projection, save in place.
    Drop-in replacement for the old flat alpha_composite approach.
    """
    mesh      = parse_obj(obj_path)
    base_img  = Image.open(texture_path).convert("RGBA")
    decal_img = Image.open(decal_path).convert("RGBA")
    result    = apply_decal_3d(base_img, decal_img, mesh, biome)
    result.save(texture_path)
