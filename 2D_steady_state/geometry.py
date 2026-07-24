import numpy as np
import torch
import config

# Global Domain 
def globalDomain(params):
    LEN_PARENT  = float(params.LEN_PARENT)
    LEN_BRANCH  = float(params.LEN_BRANCH)
    α1          = float(params.α1)
    α2          = float(params.α2)
    R0          = float(params.R0)
    R1          = float(params.R1)
    R2          = float(params.R2)
    # - 
    xmax        = LEN_PARENT + max(LEN_BRANCH * np.cos(α1), LEN_BRANCH * np.cos(α2)) + 2 * R0
    xmin        = -2 * R0
    ymax        = max(LEN_BRANCH * np.sin(α1), 0.0) + 2 * R0
    ymin        = -max(LEN_BRANCH * np.sin(α2), 0.0) - 2 * R0
    # - 
    return xmin, xmax, ymin, ymax

# Bifurcation Domain Mask
def insideBifurcationMask(coll_array, params):
    in_parent   = insideRect(coll_array=coll_array, position=(0, 0), angle=0, width=params.LEN_PARENT, height=params.R0 * 2)
    in_top      = insideRect(coll_array=coll_array, position=(params.LEN_PARENT, 0), angle=params.α1, width=params.LEN_BRANCH, height=params.R1 * 2)
    in_bottom   = insideRect(coll_array=coll_array, position=(params.LEN_PARENT, 0), angle=-params.α2, width=params.LEN_BRANCH, height=params.R2 * 2)
    # - 
    return in_parent | in_top | in_bottom

# Inside Rect Mask
def insideRect(coll_array, position, angle, width, height):
    device, dtype   = coll_array.device, coll_array.dtype
    # - 
    rect_pos        = torch.as_tensor(position, device=device, dtype=dtype)
    θ               = torch.as_tensor(angle, device=device, dtype=dtype)
    # - 
    u               = torch.stack([torch.cos(θ), torch.sin(θ)])  # unit axis along centerline
    n               = torch.stack([-torch.sin(θ), torch.cos(θ)])  # unit normal to centerline
    L               = torch.as_tensor(width, device=device, dtype=dtype)
    r               = 0.5 * torch.as_tensor(height, device=device, dtype=dtype)
    # - 
    dist_vect       = coll_array - rect_pos  # distance vector from rect origin to each point
    s               = dist_vect @ u  # dot product -> distance component along centerline
    t               = dist_vect @ n  # dot product -> distance conponent normal to centerline
    # - 
    inside_axial    = (s >= 0.0) & (s <= L)
    inside_radial   = torch.abs(t) <= r
    # - 
    return inside_axial & inside_radial

# Build Interior Collocation Points Array
def buildInteriorArray(params, device):
    xmin, xmax, ymin, ymax  = globalDomain(params)
    # - 
    x                       = torch.rand(config.N_PTS_PROPOSAL, 1, device=device, dtype=torch.float32) * (xmax - xmin) + xmin
    y                       = torch.rand(config.N_PTS_PROPOSAL, 1, device=device, dtype=torch.float32) * (ymax - ymin) + ymin
    # - 
    coll_array_raw          = torch.cat([x, y], dim=1)
    inside_geometry_mask    = insideBifurcationMask(coll_array_raw, params)
    coll_array              = coll_array_raw[inside_geometry_mask]
    max_radius              = torch.max(torch.stack([params.R0, params.R1, params.R2]))
    centerline_skewed       = skewTowardCenterline(
        points=coll_array,
        radius=config.CENTERLINE_SKEW_RADIUS_FACTOR * max_radius,
        strength=config.CENTERLINE_SKEW_STRENGTH,
    )
    centerline_skewed_inside = insideBifurcationMask(centerline_skewed, params)
    coll_array              = torch.where(centerline_skewed_inside.unsqueeze(1), centerline_skewed, coll_array)
    coll_array              = skewTowardInnerWalls(
        points=coll_array,
        params=params,
        strength=config.INNER_WALL_SKEW_STRENGTH,
        influence_radius=config.INNER_WALL_SKEW_RADIUS_FACTOR * max_radius,
    )
    # - 
    return coll_array

def sampleInteriorPoints(params, device, n_points, dtype=torch.float32, proposal_factor=8, max_rounds=32):
    n_points = int(n_points)
    if n_points <= 0:
        return torch.empty((0, 2), device=device, dtype=dtype)

    xmin, xmax, ymin, ymax = globalDomain(params)
    proposal_factor = max(float(proposal_factor), 1.0)
    proposal_batch = max(int(np.ceil(n_points * proposal_factor)), 1024)
    remaining = n_points
    chunks = []

    for _ in range(int(max_rounds)):
        raw_n = max(proposal_batch, int(np.ceil(remaining * proposal_factor)))
        x = torch.rand(raw_n, 1, device=device, dtype=dtype) * (xmax - xmin) + xmin
        y = torch.rand(raw_n, 1, device=device, dtype=dtype) * (ymax - ymin) + ymin
        raw = torch.cat([x, y], dim=1)
        inside = raw[insideBifurcationMask(raw, params)]

        if inside.shape[0] > 0:
            take = min(remaining, inside.shape[0])
            chunks.append(inside[:take])
            remaining -= take
            if remaining == 0:
                return torch.cat(chunks, dim=0)

        proposal_batch = int(np.ceil(proposal_batch * 1.25))

    raise RuntimeError(f"Could only sample {n_points - remaining} of {n_points} requested interior points.")

def skewTowardCenterline(points, radius, strength=0.5):
    """Pull points toward the global horizontal centerline at y = 0."""
    if points.shape[0] == 0:
        return points
    if not 0.0 <= float(strength) < 1.0:
        raise ValueError("strength must satisfy 0 <= strength < 1.")
    radius = torch.as_tensor(radius, device=points.device, dtype=points.dtype)
    if radius <= 0.0:
        raise ValueError("radius must be positive.")

    y = points[:, 1:2]
    weight = torch.exp(-0.5 * (y / radius)**2)
    skewed = points.clone()
    skewed[:, 1:2] = y * (1.0 - float(strength) * weight)
    return skewed

def skewTowardInnerWalls(points, params, strength=0.65, influence_radius=None):
    """Pull interior points smoothly toward the two walls meeting at the V-corner."""
    if points.shape[0] == 0:
        return points
    if not 0.0 <= float(strength) < 1.0:
        raise ValueError("strength must satisfy 0 <= strength < 1.")

    device, dtype = points.device, points.dtype
    walls = bifurcationWalls(params=params, device=device, dtype=dtype)
    v_corner = lineIntersection(walls["top_minus"], walls["bottom_plus"], dtype=dtype)

    # Only the portions downstream of the V-corner are exterior inner walls.
    inner_segments = (
        (v_corner, walls["top_minus"]["x1"]),
        (v_corner, walls["bottom_plus"]["x1"]),
    )
    distances = []
    closest_points = []
    for x0, x1 in inner_segments:
        distance, closest, _ = pointToSegmentDistance(points=points, x0=x0, x1=x1)
        distances.append(distance)
        closest_points.append(closest)

    distance_by_wall = torch.cat(distances, dim=1)
    closest_by_wall = torch.stack(closest_points, dim=1)
    nearest_distance, nearest_wall = torch.min(distance_by_wall, dim=1, keepdim=True)
    gather_index = nearest_wall.unsqueeze(2).expand(-1, 1, 2)
    target = torch.gather(closest_by_wall, dim=1, index=gather_index).squeeze(1)

    if influence_radius is None:
        influence_radius = 2.0 * torch.max(torch.stack([params.R0, params.R1, params.R2]))
    radius = torch.as_tensor(influence_radius, device=device, dtype=dtype)
    if radius <= 0.0:
        raise ValueError("influence_radius must be positive.")

    weight = torch.exp(-0.5 * (nearest_distance / radius)**2)
    skewed = points + float(strength) * weight * (target - points)

    # A straight pull toward a non-convex boundary can leave the union of branch
    # rectangles. Preserve the original point in that exceptional case.
    remains_inside = insideBifurcationMask(skewed, params)
    return torch.where(remains_inside.unsqueeze(1), skewed, points)

# Rect Wall Segments 
def rectWalls(position, angle, width, height, device, dtype):
        pos         = torch.as_tensor(position, device=device, dtype=dtype)
        theta       = torch.as_tensor(angle, device=device, dtype=dtype)
        # -
        u           = torch.stack([torch.cos(theta), torch.sin(theta)])   # points towards out_wall
        n           = torch.stack([-torch.sin(theta), torch.cos(theta)])  # points towards top_wall
        L           = torch.as_tensor(width, device=device, dtype=dtype)  # length
        r           = 0.5 * torch.as_tensor(height, device=device, dtype=dtype)  # radius
        # -
        a           = pos  # start position (at in_wall on centerline)
        b           = pos + L * u  # end positon (at out_wall on centerline)
        # -
        top_wall    = {"x0": a + r * n, "x1": b + r * n, "normal": n}
        bottom_wall = {"x0": a - r * n, "x1": b - r * n, "normal": -n}
        in_wall     = {"x0": a - r * n, "x1": a + r * n, "normal": -u}
        out_wall    = {"x0": b - r * n, "x1": b + r * n, "normal": u}
        # - 
        return top_wall, bottom_wall, in_wall, out_wall

# Bifurcation Wall Segments 
def bifurcationWalls(params, device, dtype):
    prnt_top_wall, prnt_bottom_wall, prnt_in_wall, prnt_out_wall = rectWalls(position=(0.0, 0.0), angle=0.0, width=params.LEN_PARENT, height=params.R0 * 2, device=device, dtype=dtype)
    bch1_top_wall, bch1_bottom_wall, bch1_in_wall, bch1_out_wall = rectWalls(position=(params.LEN_PARENT, 0.0), angle=params.α1, width=params.LEN_BRANCH, height=params.R1 * 2, device=device, dtype=dtype)
    bch2_top_wall, bch2_bottom_wall, bch2_in_wall, bch2_out_wall = rectWalls(position=(params.LEN_PARENT, 0.0), angle=-params.α2, width=params.LEN_BRANCH, height=params.R2 * 2, device=device, dtype=dtype)
    # -
    segments = {  # includes x0, x1, and normal directions for each segment
        "parent_plus": prnt_top_wall,
        "parent_minus": prnt_bottom_wall,
        "top_plus": bch1_top_wall,
        "top_minus": bch1_bottom_wall,
        "bottom_plus": bch2_top_wall,
        "bottom_minus": bch2_bottom_wall,
        # - 
        "inlet": prnt_in_wall,
        "outlet_top": bch1_out_wall,
        "outlet_bottom": bch2_out_wall,
    }
    return segments

def lineIntersection(first, second, dtype):
    p = first["x0"]
    r = first["x1"] - first["x0"]
    q = second["x0"]
    s = second["x1"] - second["x0"]

    def cross2(a, b):
        return a[0] * b[1] - a[1] * b[0]

    denom = cross2(r, s)
    parallel_tol = 10.0 * torch.finfo(dtype).eps * torch.linalg.norm(r) * torch.linalg.norm(s)
    if torch.abs(denom) <= parallel_tol:
        raise ValueError("Could not locate the inner bifurcation V corner.")

    return p + (cross2(q - p, s) / denom) * r

def innerBifurcationVCorner(params, device, dtype):
    walls = bifurcationWalls(params=params, device=device, dtype=dtype)
    return lineIntersection(walls["top_minus"], walls["bottom_plus"], dtype=dtype)

# Build Walls Collocation Points Array
def buildWallsArray(segments, n_per_segment):
    names = list(segments.keys())
    # -
    N_PTS_wall_dict = {name: n_per_segment for name in names}
    # -
    points_all = []
    normals_all = []
    sid_all = []
    pts_by_segment = {}
    normals_by_segment = {}
    # -
    for i, name in enumerate(names):
        N_PTS = N_PTS_wall_dict[name]
        # -
        segment = segments[name]
        x0 = segment["x0"]
        x1 = segment["x1"]
        # -
        t = torch.linspace(0.0, 1.0, N_PTS, device=x0.device, dtype=x0.dtype)
        # -
        pts = x0.unsqueeze(0) + t.unsqueeze(1) * (x1 - x0).unsqueeze(0)
        normal = segment["normal"].unsqueeze(0).repeat(N_PTS, 1)
        # -
        pts_by_segment[name] = pts
        normals_by_segment[name] = normal
        points_all.append(pts)
        normals_all.append(normal)
        sid_all.append(torch.full((N_PTS,), i, device=x0.device, dtype=torch.long))
    # - 
    points = torch.cat(points_all, dim=0)
    normals = torch.cat(normals_all, dim=0)
    segment_ID = torch.cat(sid_all, dim=0)
    # -
    out = {
        "segment_names": names,
        "points_by_segment": pts_by_segment,
        "normals_by_segment": normals_by_segment,
        "points": points,
        "normals": normals,
        "segment_id": segment_ID,
    }
    return out

# Wall Extends Inside Bifurcation Mask
def exteriorBoundaryMask(points, normals, params, eps):
    p_plus          = points + eps * normals  # check slightly above wall
    p_minus         = points - eps * normals  # check slightly below wall
    # -
    inside_plus     = insideBifurcationMask(p_plus, params)
    inside_minus    = insideBifurcationMask(p_minus, params)
    # - 
    return inside_plus ^ inside_minus  # false if both checks are inside the bifurcation (wall is inside bifurcation)

# Keep Valid Wall Collocation Points Only
def keepExteriorWallPoints(wall, params, eps):
    points      = wall["points"]
    normals     = wall["normals"]
    segment_id  = wall["segment_id"]
    names       = wall["segment_names"]
    # -
    keep        = exteriorBoundaryMask(points=points, normals=normals, params=params, eps=eps)
    points      = points[keep]
    normals     = normals[keep]
    segment_id  = segment_id[keep]
    # -
    points_by_segment = {}
    normals_by_segment = {}
    for i, name in enumerate(names):
        m = segment_id == i
        points_by_segment[name] = points[m]
        normals_by_segment[name] = normals[m]
    # -
    out = {
        "segment_names": names,
        "points_by_segment": points_by_segment,
        "normals_by_segment": normals_by_segment,
        "points": points,
        "normals": normals,
        "segment_id": segment_id,
    }
    return out

def addInnerVCornerWallPoint(wall, params, segment_name="top_minus"):
    names = wall["segment_names"]
    if segment_name not in names:
        raise ValueError(f"Could not attach V-corner wall point to missing segment '{segment_name}'.")

    points = wall["points"]
    dtype = points.dtype
    device = points.device
    v_corner = innerBifurcationVCorner(params=params, device=device, dtype=dtype)
    rmax = torch.stack([params.R0, params.R1, params.R2]).to(device=device, dtype=dtype).max()
    duplicate_tol = torch.clamp(10.0 * torch.finfo(dtype).eps * rmax, min=torch.finfo(dtype).eps)
    already_present = torch.any(torch.linalg.norm(points - v_corner.unsqueeze(0), dim=1) <= duplicate_tol)
    if bool(already_present):
        return wall

    segment_id = names.index(segment_name)
    normal = wall["normals_by_segment"][segment_name][0]

    points = torch.cat([points, v_corner.unsqueeze(0)], dim=0)
    normals = torch.cat([wall["normals"], normal.unsqueeze(0)], dim=0)
    segment_ids = torch.cat(
        [
            wall["segment_id"],
            torch.tensor([segment_id], device=device, dtype=torch.long),
        ],
        dim=0,
    )

    points_by_segment = dict(wall["points_by_segment"])
    normals_by_segment = dict(wall["normals_by_segment"])
    points_by_segment[segment_name] = torch.cat(
        [points_by_segment[segment_name], v_corner.unsqueeze(0)],
        dim=0,
    )
    normals_by_segment[segment_name] = torch.cat(
        [normals_by_segment[segment_name], normal.unsqueeze(0)],
        dim=0,
    )

    out = dict(wall)
    out.update({
        "points_by_segment": points_by_segment,
        "normals_by_segment": normals_by_segment,
        "points": points,
        "normals": normals,
        "segment_id": segment_ids,
    })
    return out

# Extract Wall Segments
def exteriorWallSegments(segments, params, eps, n_samples=1000, side_walls_only=False):
    if side_walls_only: names = [name for name in segments.keys() if ("plus" in name or "minus" in name)]
    else: names = list(segments.keys())
    # -
    out = {}
    for name in names:
        segment = segments[name]
        x0 = segment["x0"]
        x1 = segment["x1"]
        normal = segment["normal"]

        t = torch.linspace(0.0, 1.0, int(n_samples), device=x0.device, dtype=x0.dtype)
        points = x0.unsqueeze(0) + t.unsqueeze(1) * (x1 - x0).unsqueeze(0)
        normals = normal.unsqueeze(0).repeat(points.shape[0], 1)
        keep = exteriorBoundaryMask(points=points, normals=normals, params=params, eps=eps)
        keep_idx = torch.nonzero(keep, as_tuple=False).reshape(-1)
        if keep_idx.numel() < 2: continue

        run_start = int(keep_idx[0].item())
        run_prev = run_start
        run_number = 0

        def append_run(start_idx, end_idx, suffix):
            if end_idx <= start_idx:
                return
            p0 = points[start_idx]
            p1 = points[end_idx]
            if torch.linalg.norm(p1 - p0) <= eps:
                return
            out[f"{name}_ext{suffix}"] = {
                "x0": p0,
                "x1": p1,
                "normal": normal,
                "source_name": name,
            }

        for idx_tensor in keep_idx[1:]:
            idx = int(idx_tensor.item())
            if idx != run_prev + 1:
                append_run(run_start, run_prev, run_number)
                run_number += 1
                run_start = idx
            run_prev = idx
        append_run(run_start, run_prev, run_number)

    return out

# Nearest Boundary Direction
def distanceToNearestBoundaryPoints(points, boundary_points, eps=1e-12):

    if boundary_points.shape[0] == 0:
        dist_min = torch.full(
            (points.shape[0], 1),
            float("inf"),
            device=points.device,
            dtype=points.dtype,
        )
        idx_min = torch.full(
            (points.shape[0],),
            -1,
            device=points.device,
            dtype=torch.long,
        )
        unit_dir_min = torch.zeros(
            (points.shape[0], 2),
            device=points.device,
            dtype=points.dtype,
        )
        return dist_min, idx_min, unit_dir_min

    delta = boundary_points.unsqueeze(0) - points.unsqueeze(1)   # [N, M, 2]
    dist_sq = torch.sum(delta * delta, dim=2).clamp_min(eps)     # [N, M]

    idx_min = torch.argmin(dist_sq, dim=1)                       # [N]
    dist_min = torch.sqrt(dist_sq.gather(1, idx_min.unsqueeze(1)))  # [N, 1]

    gather_idx = idx_min.view(-1, 1, 1).expand(-1, 1, 2)
    delta_min = torch.gather(delta, dim=1, index=gather_idx).squeeze(1)  # [N, 2]
    unit_dir_min = delta_min / (dist_min + eps)                           # [N, 2]

    return dist_min, idx_min, unit_dir_min

# Geometry Compute Dtype/Device (config.WALL_DISTANCE_DTYPE; MPS has no float64 support,
# so fall back to CPU when float64 is requested there)
def _geometryComputeDtypeAndDevice(device):
    compute_dtype = getattr(config, "WALL_DISTANCE_DTYPE", torch.float64)
    if compute_dtype == torch.float64 and device.type == "mps":
        return compute_dtype, torch.device("cpu")
    return compute_dtype, device

# Point --> Segment Distance
def pointToSegmentDistance(points, x0, x1, eps=1e-24):
    # Precision is set by config.WALL_DISTANCE_DTYPE: at these coordinate magnitudes
    # (~1e-4 m), float32's own rounding error (~1e-11 m) swamps an eps this small,
    # turning the near-wall floor into noise instead of a controlled regularizer.
    orig_dtype, orig_device = points.dtype, points.device
    compute_dtype, compute_device = _geometryComputeDtypeAndDevice(orig_device)
    points_c = points.to(device=compute_device, dtype=compute_dtype)
    x0_c = x0.to(device=compute_device, dtype=compute_dtype)
    x1_c = x1.to(device=compute_device, dtype=compute_dtype)
    # -
    seg = x1_c - x0_c
    seg_len_sq = torch.sum(seg * seg).clamp_min(eps)
    t = ((points_c - x0_c) @ seg) / seg_len_sq
    t = torch.clamp(t, 0.0, 1.0).unsqueeze(1)
    closest = x0_c.unsqueeze(0) + t * seg.unsqueeze(0)
    delta = closest - points_c
    dist = torch.sqrt(torch.sum(delta * delta, dim=1, keepdim=True) + eps)  # check eps = 0.0
    unit_to_segment = delta / (dist + eps)
    return (
        dist.to(device=orig_device, dtype=orig_dtype),
        closest.to(device=orig_device, dtype=orig_dtype),
        unit_to_segment.to(device=orig_device, dtype=orig_dtype),
    )

# Point --> Segment Normal Projection
def pointToSegmentNormalProjection(points, x0, x1, normal, eps=1e-24):
    # Precision is set by config.WALL_DISTANCE_DTYPE; see pointToSegmentDistance for why.
    orig_dtype, orig_device = points.dtype, points.device
    compute_dtype, compute_device = _geometryComputeDtypeAndDevice(orig_device)
    points_c = points.to(device=compute_device, dtype=compute_dtype)
    x0_c = x0.to(device=compute_device, dtype=compute_dtype)
    x1_c = x1.to(device=compute_device, dtype=compute_dtype)
    normal_c = normal.to(device=compute_device, dtype=compute_dtype)
    # -
    seg = x1_c - x0_c
    seg_len_sq = torch.sum(seg * seg).clamp_min(eps)
    normal_c = normal_c / torch.linalg.norm(normal_c).clamp_min(eps)
    rel = points_c - x0_c

    t = (rel @ seg) / seg_len_sq
    within_projection = (t >= 0.0) & (t <= 1.0)
    signed_normal_dist = (rel @ normal_c).unsqueeze(1)
    dist = torch.sqrt(signed_normal_dist * signed_normal_dist + eps)  # check eps = 0.0

    normal_sign = torch.where(
        signed_normal_dist <= 0.0,
        torch.ones_like(signed_normal_dist),
        -torch.ones_like(signed_normal_dist),
    )
    unit_to_wall = normal_sign * normal_c.unsqueeze(0)
    return (
        dist.to(device=orig_device, dtype=orig_dtype),
        t.to(device=orig_device, dtype=orig_dtype),
        within_projection.to(device=orig_device),
        unit_to_wall.to(device=orig_device, dtype=orig_dtype),
    )

# Distance Field
def distanceToNearestSegment(points, segments, include_caps=False):
    if include_caps: names = list(segments.keys())
    else: names = [name for name in segments.keys() if ("plus" in name or "minus" in name)]
    # - 
    if len(names) == 0:
        dist_min = torch.full((points.shape[0], 1), float("inf"), device=points.device, dtype=points.dtype)
        seg_idx = torch.full((points.shape[0],), -1, device=points.device, dtype=torch.long)
        unit_dir = torch.zeros((points.shape[0], 2), device=points.device, dtype=points.dtype)
        return dist_min, seg_idx, unit_dir, names
    # - 
    dist_list = []
    unit_dir_list = []
    for name in names:
        s = segments[name]
        d, _, unit_dir = pointToSegmentDistance(points=points, x0=s["x0"], x1=s["x1"])
        dist_list.append(d)
        unit_dir_list.append(unit_dir)
    # - 
    D = torch.cat(dist_list, dim=1)
    dist_min, seg_idx = torch.min(D, dim=1, keepdim=True)
    # - 
    U = torch.stack(unit_dir_list, dim=1)  # [N, Nseg, 2]
    gather_idx = seg_idx.unsqueeze(2).expand(-1, 1, 2)
    unit_dir_min = torch.gather(U, dim=1, index=gather_idx).squeeze(1)
    # - 
    return dist_min, seg_idx.squeeze(1), unit_dir_min, names

# Distance Field Using Wall-Normal Projections With Endpoint Fallback
def distanceToNearestSegmentNormalProjection(points, segments, include_caps=False):
    if include_caps: names = list(segments.keys())
    else: names = [name for name in segments.keys() if ("plus" in name or "minus" in name)]
    # -
    if len(names) == 0:
        dist_min = torch.full((points.shape[0], 1), float("inf"), device=points.device, dtype=points.dtype)
        seg_idx = torch.full((points.shape[0],), -1, device=points.device, dtype=torch.long)
        unit_dir = torch.zeros((points.shape[0], 2), device=points.device, dtype=points.dtype)
        return dist_min, seg_idx, unit_dir, names
    # -
    nearest_dist_list = []
    nearest_unit_dir_list = []
    normal_dist_list = []
    unit_dir_list = []
    valid_list = []
    for name in names:
        s = segments[name]
        d_nearest, _, nearest_unit_dir = pointToSegmentDistance(points=points, x0=s["x0"], x1=s["x1"])
        d_normal, _, valid, unit_dir = pointToSegmentNormalProjection(
            points=points,
            x0=s["x0"],
            x1=s["x1"],
            normal=s["normal"],
        )
        nearest_dist_list.append(d_nearest)
        nearest_unit_dir_list.append(nearest_unit_dir)
        normal_dist_list.append(d_normal)
        unit_dir_list.append(unit_dir)
        valid_list.append(valid)
    # -
    D_nearest = torch.cat(nearest_dist_list, dim=1)
    _, seg_idx = torch.min(D_nearest, dim=1, keepdim=True)
    # -
    D_normal = torch.cat(normal_dist_list, dim=1)
    V = torch.stack(valid_list, dim=1)
    U = torch.stack(unit_dir_list, dim=1)
    U_nearest = torch.stack(nearest_unit_dir_list, dim=1)
    # -
    gather_dist_idx = seg_idx
    gather_vec_idx = seg_idx.unsqueeze(2).expand(-1, 1, 2)
    selected_valid = torch.gather(V, dim=1, index=gather_dist_idx).squeeze(1)
    nearest_dist = torch.gather(D_nearest, dim=1, index=gather_dist_idx)
    nearest_unit_dir = torch.gather(U_nearest, dim=1, index=gather_vec_idx).squeeze(1)
    dist_min = torch.gather(D_normal, dim=1, index=gather_dist_idx)
    unit_dir_min = torch.gather(U, dim=1, index=gather_vec_idx).squeeze(1)
    # -
    # At segment endpoints/corners, use the closest wall point's radial
    # direction instead of masking lift out as a singular projection.
    dist_min = torch.where(
        selected_valid.unsqueeze(1),
        dist_min,
        nearest_dist,
    )
    unit_dir_min = torch.where(
        selected_valid.unsqueeze(1),
        unit_dir_min,
        nearest_unit_dir,
    )
    # -
    return dist_min, seg_idx.squeeze(1), unit_dir_min, names

# Build Bifurcation Array
def buildBifurcationArray(params, device, n_boundary_per_segment=config.N_PTS_BDR):
    rmax = torch.stack([params.R0, params.R1, params.R2]).max().detach().cpu().item()
    boundary_eps = 1e-3 * rmax
    wall_segment_clip_samples = int(getattr(config, "WALL_SEGMENT_CLIP_SAMPLES", max(400, 20 * n_boundary_per_segment)))

    # Interior Array, Wall Array, & Full Array
    interior_array = buildInteriorArray(params=params, device=device)
    # - 
    walls = bifurcationWalls(params=params, device=device, dtype=interior_array.dtype)
    bifurcation_boundary = buildWallsArray(segments=walls, n_per_segment=n_boundary_per_segment)
    bifurcation_boundary = keepExteriorWallPoints(wall=bifurcation_boundary, params=params, eps=boundary_eps)
    bifurcation_boundary = addInnerVCornerWallPoint(wall=bifurcation_boundary, params=params)
    boundary_array = bifurcation_boundary["points"]
    # - 
    full_array = torch.cat([interior_array, boundary_array], dim=0)
    
    # Extract Wall & Boundary Segments
    side_wall_segments = exteriorWallSegments(
        segments=walls,
        params=params,
        eps=boundary_eps,
        n_samples=wall_segment_clip_samples,
        side_walls_only=True,
    )
    boundary_segments = exteriorWallSegments(
        segments=walls,
        params=params,
        eps=boundary_eps,
        n_samples=wall_segment_clip_samples,
        side_walls_only=False,
    )

    # Wall Distance Filed
    dist_to_wall, nearest_wall_id, dir_to_wall, wall_names_only = distanceToNearestSegmentNormalProjection(
            points=full_array,
            segments=side_wall_segments,
            include_caps=True,
        )

    # Boundary Distance Field
    dist_to_boundary, nearest_boundary_id, dir_to_boundary, boundary_names_all = distanceToNearestSegment(
            points=full_array,
            segments=boundary_segments,
            include_caps=True,
        )

    # Return Geometry Data
    n_coll = interior_array.shape[0]
    n_full = full_array.shape[0]
    # - 
    is_boundary = torch.zeros(n_full, device=device, dtype=torch.bool)
    is_boundary[n_coll:] = True
    is_interior = ~is_boundary
    # - 
    full_segment_id = torch.full((n_full,), -1, device=device, dtype=torch.long)
    full_segment_id[n_coll:] = bifurcation_boundary["segment_id"]
    # - 
    return {
        "interior_array": interior_array,
        "boundary_array": boundary_array,
        "boundary_segment_id": bifurcation_boundary["segment_id"],
        "full_array": full_array,
        "is_interior": is_interior,
        "is_boundary": is_boundary,
        "full_segment_id": full_segment_id,
        "segment_names": bifurcation_boundary["segment_names"],
        "boundary_normals": bifurcation_boundary["normals"],
        "points_by_segment": bifurcation_boundary["points_by_segment"],
        "normals_by_segment": bifurcation_boundary["normals_by_segment"],
        "distance_to_nearest_wall": dist_to_wall,
        "nearest_wall_id": nearest_wall_id,
        "direction_to_nearest_wall": dir_to_wall,
        "wall_names_only": wall_names_only,
        "side_wall_segments": side_wall_segments,
        "distance_to_nearest_boundary": dist_to_boundary,
        "nearest_boundary_id": nearest_boundary_id,
        "direction_to_nearest_boundary": dir_to_boundary,
        "boundary_names_all": boundary_names_all,
        "boundary_segments": boundary_segments,
    }

def rebuildBifurcationArrayWithInterior(params, device, array, interior_array):
    dtype = array["full_array"].dtype
    interior_array = interior_array.detach().to(device=device, dtype=dtype)
    boundary_array = array["boundary_array"].detach().to(device=device, dtype=dtype)
    full_array = torch.cat([interior_array, boundary_array], dim=0)

    side_wall_segments = array["side_wall_segments"]
    boundary_segments = array["boundary_segments"]

    dist_to_wall, nearest_wall_id, dir_to_wall, wall_names_only = distanceToNearestSegmentNormalProjection(
        points=full_array,
        segments=side_wall_segments,
        include_caps=True,
    )
    dist_to_boundary, nearest_boundary_id, dir_to_boundary, boundary_names_all = distanceToNearestSegment(
        points=full_array,
        segments=boundary_segments,
        include_caps=True,
    )

    n_coll = interior_array.shape[0]
    n_full = full_array.shape[0]
    is_boundary = torch.zeros(n_full, device=device, dtype=torch.bool)
    is_boundary[n_coll:] = True
    is_interior = ~is_boundary

    boundary_segment_id = array["boundary_segment_id"].detach().to(device=device)
    full_segment_id = torch.full((n_full,), -1, device=device, dtype=torch.long)
    full_segment_id[n_coll:] = boundary_segment_id

    out = dict(array)
    out.update({
        "interior_array": interior_array,
        "boundary_array": boundary_array,
        "boundary_segment_id": boundary_segment_id,
        "full_array": full_array,
        "is_interior": is_interior,
        "is_boundary": is_boundary,
        "full_segment_id": full_segment_id,
        "distance_to_nearest_wall": dist_to_wall,
        "nearest_wall_id": nearest_wall_id,
        "direction_to_nearest_wall": dir_to_wall,
        "wall_names_only": wall_names_only,
        "distance_to_nearest_boundary": dist_to_boundary,
        "nearest_boundary_id": nearest_boundary_id,
        "direction_to_nearest_boundary": dir_to_boundary,
        "boundary_names_all": boundary_names_all,
    })
    return out

def nearestSideWallGeometry(points, params, array, device, dtype):
    del params
    side_wall_segments = array.get("side_wall_segments")
    if side_wall_segments is not None and len(side_wall_segments) > 0:
        dist_to_wall, _, dir_to_wall, _ = distanceToNearestSegmentNormalProjection(
            points=points,
            segments=side_wall_segments,
            include_caps=True,
        )
        return dist_to_wall.to(dtype=dtype), dir_to_wall.to(dtype=dtype)

    side_wall_names = [name for name in array["segment_names"] if ("plus" in name or "minus" in name)]
    side_wall_points = [array["points_by_segment"][name] for name in side_wall_names if array["points_by_segment"][name].shape[0] > 0]
    if len(side_wall_points) == 0:
        raise ValueError("No side-wall points available for lift-force evaluation.")

    side_wall_points = torch.cat(side_wall_points, dim=0).to(device=device, dtype=dtype)
    dist_to_wall, _, dir_to_wall = distanceToNearestBoundaryPoints(
        points=points,
        boundary_points=side_wall_points,
    )
    return dist_to_wall.to(dtype=dtype), dir_to_wall.to(dtype=dtype)
