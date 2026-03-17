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
    pos             = torch.as_tensor(position, device=device, dtype=dtype)
    theta           = torch.as_tensor(angle, device=device, dtype=dtype)
    # - 
    u               = torch.stack([torch.cos(theta), torch.sin(theta)])  # unit axis
    n               = torch.stack([-torch.sin(theta), torch.cos(theta)]) # unit normal
    L               = torch.as_tensor(width, device=device, dtype=dtype)
    r               = 0.5 * torch.as_tensor(height, device=device, dtype=dtype)
    # - 
    A               = pos
    # - 
    AP              = coll_array - A
    s               = AP @ u
    inside_axial    = (s >= 0.0) & (s <= L)
    t               = AP @ n
    inside_radial   = torch.abs(t) <= r
    # - 
    return inside_axial & inside_radial

# Build Interior Collocation Points Array
def buildInteriorArray(params, device):
    xmin, xmax, ymin, ymax  = globalDomain(params)
    # - 
    x                       = torch.rand(config.N_PTS, 1, device=device, dtype=torch.float32) * (xmax - xmin) + xmin
    y                       = torch.rand(config.N_PTS, 1, device=device, dtype=torch.float32) * (ymax - ymin) + ymin
    # - 
    coll_array_raw          = torch.cat([x, y], dim=1)
    inside_geometry_mask    = insideBifurcationMask(coll_array_raw, params)
    coll_array              = coll_array_raw[inside_geometry_mask]
    # - 
    return coll_array

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
    segments = {
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

# Build Walls Collocation Points Array
def buildWallsArray(segments, n_per_segment, include_endpoints=False):
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
        if include_endpoints: t = torch.linspace(0.0, 1.0, N_PTS, device=x0.device, dtype=x0.dtype)
        else: t = torch.linspace(0.0, 1.0, N_PTS + 2, device=x0.device, dtype=x0.dtype)[1:-1]
        # -
        pts = x0.unsqueeze(0) + t.unsqueeze(1) * (x1 - x0).unsqueeze(0)
        normal = segment["normal"].unsqueeze(0).repeat(N_PTS, 1)
        # -
        pts_by_segment[name] = pts
        normals_by_segment[name] = normal
        points_all.append(pts)
        normals_all.append(normal)
        sid_all.append(torch.full((N_PTS,), i, device=x0.device, dtype=torch.long))

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

# Distance/closest-point data from each point to a finite wall segment.
def pointToSegmentDistance(points, x0, x1, eps=1e-12):
    seg = x1 - x0
    seg_len_sq = torch.sum(seg * seg).clamp_min(eps)
    t = ((points - x0) @ seg) / seg_len_sq
    t = torch.clamp(t, 0.0, 1.0).unsqueeze(1)
    closest = x0.unsqueeze(0) + t * seg.unsqueeze(0)
    delta = closest - points
    dist = torch.sqrt(torch.sum(delta * delta, dim=1, keepdim=True) + eps)
    unit_to_segment = delta / (dist + eps)
    return dist, closest, unit_to_segment

def distanceToNearestSegment(points, segments, include_caps=False):
    if include_caps:
        names = list(segments.keys())
    else:
        names = [name for name in segments.keys() if ("plus" in name or "minus" in name)]

    if len(names) == 0:
        dist_min = torch.full((points.shape[0], 1), float("inf"), device=points.device, dtype=points.dtype)
        seg_idx = torch.full((points.shape[0],), -1, device=points.device, dtype=torch.long)
        unit_dir = torch.zeros((points.shape[0], 2), device=points.device, dtype=points.dtype)
        return dist_min, seg_idx, unit_dir, names

    dist_list = []
    unit_dir_list = []
    for name in names:
        s = segments[name]
        d, _, unit_dir = pointToSegmentDistance(points=points, x0=s["x0"], x1=s["x1"])
        dist_list.append(d)
        unit_dir_list.append(unit_dir)

    D = torch.cat(dist_list, dim=1)
    dist_min, seg_idx = torch.min(D, dim=1, keepdim=True)

    U = torch.stack(unit_dir_list, dim=1)  # [N, Nseg, 2]
    gather_idx = seg_idx.unsqueeze(2).expand(-1, 1, 2)
    unit_dir_min = torch.gather(U, dim=1, index=gather_idx).squeeze(1)
    return dist_min, seg_idx.squeeze(1), unit_dir_min, names

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

# Build Bifurcation Array
def buildBifurcationArray(params, device, n_boundary_per_segment=config.N_PTS_BDR, include_endpoints=False):
    rmax = torch.stack([params.R0, params.R1, params.R2]).max().detach().cpu().item()
    boundary_eps = 1e-3 * rmax
    # - 
    interior_array          = buildInteriorArray(params=params, device=device)
    walls                   = bifurcationWalls(params=params, device=device, dtype=interior_array.dtype)
    bifurcation_boundary    = buildWallsArray(segments=walls, n_per_segment=n_boundary_per_segment, include_endpoints=include_endpoints)
    bifurcation_boundary    = keepExteriorWallPoints(wall=bifurcation_boundary, params=params, eps=boundary_eps)
    # - 
    boundary_array          = bifurcation_boundary["points"]
    full_array              = torch.cat([interior_array, boundary_array], dim=0)
    # Distance fields on full geometry points.
    dist_to_wall, nearest_wall_id, dir_to_wall, wall_names = distanceToNearestSegment(
        points=full_array,
        segments=walls,
        include_caps=False,
    )
    dist_to_boundary, nearest_boundary_id, dir_to_boundary, boundary_names = distanceToNearestSegment(
        points=full_array,
        segments=walls,
        include_caps=True,
    )
    # -
    n_coll                  = interior_array.shape[0]
    n_full                  = full_array.shape[0]
    # -
    is_boundary             = torch.zeros(n_full, device=device, dtype=torch.bool)  # initialize all false 
    is_boundary[n_coll:]    = True  # first rows only are boundary
    is_interior             = ~is_boundary  # interior points
    # -
    full_segment_id         = torch.full((n_full,), -1, device=device, dtype=torch.long)
    full_segment_id[n_coll:]= bifurcation_boundary["segment_id"]
    # -
    out = {
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
        "wall_names_only": wall_names,
        "distance_to_nearest_boundary": dist_to_boundary,
        "nearest_boundary_id": nearest_boundary_id,
        "direction_to_nearest_boundary": dir_to_boundary,
        "boundary_names_all": boundary_names,
    }
    # -
    return out
