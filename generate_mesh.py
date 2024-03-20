from sys import argv
import numpy as np
import gmsh

def build_two_phase_mesh(bbox:np.ndarray, markers:np.ndarray, h_size, dist_max = 1) -> None:
    """
    Build the two-phase mesh, given the interface markers. 
    bbox    [in, 2 x 2]  [[x_lo, y_lo], [x_hi, y_hi]]
    markers [in, Np x 2] with the orientation where the outward normal is pointing to fluid 2. 
    h_size  [in, 2] [h_min, h_max]
    dist_max [in, 1] distance from interface where h_max is attained
    """
    gmsh.initialize()
    gmsh.model.add("two-phase")
    # write the points
    num_markers = markers.shape[0]
    pts_ll = gmsh.model.geo.addPoint(bbox[0,0], bbox[0,1], 0) # lower left corner
    pts_lr = gmsh.model.geo.addPoint(bbox[1,0], bbox[0,1], 0) # lower right corner
    pts_ur = gmsh.model.geo.addPoint(bbox[1,0], bbox[1,1], 0) # upper right corner
    pts_ul = gmsh.model.geo.addPoint(bbox[0,0], bbox[1,1], 0) # upper left corner
    pts_marker = [gmsh.model.geo.addPoint(markers[i,0], markers[i,1], 0) for i in range(num_markers)]

    # add the line segments
    e_dry_l = gmsh.model.geo.addLine(pts_ll, pts_marker[-1])
    e_wet = gmsh.model.geo.addLine(pts_marker[-1], pts_marker[0])
    e_dry_r = gmsh.model.geo.addLine(pts_marker[0], pts_lr)
    e_right = gmsh.model.geo.addLine(pts_lr, pts_ur)
    e_top = gmsh.model.geo.addLine(pts_ur, pts_ul)
    e_left = gmsh.model.geo.addLine(pts_ul, pts_ll)
    e_marker = [gmsh.model.geo.addLine(pts_marker[i], pts_marker[i+1]) for i in range(num_markers-1)]
    # add the line loop
    loop_1 = gmsh.model.geo.addCurveLoop([e_wet] + e_marker)
    fluid_1 = gmsh.model.geo.addPlaneSurface([loop_1])
    loop_2 = gmsh.model.geo.addCurveLoop([e_dry_l] + [-e for e in reversed(e_marker)] + [e_dry_r, e_right, e_top, e_left])
    fluid_2 = gmsh.model.geo.addPlaneSurface([loop_2])
    gmsh.model.geo.synchronize()

    # add the mesh size fields
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", e_marker)
    # gmsh.model.mesh.field.setNumber(1, "Sampling", 20)

    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", h_size[0])
    gmsh.model.mesh.field.setNumber(2, "SizeMax", h_size[1])
    gmsh.model.mesh.field.setNumber(2, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(2, "DistMax", dist_max)

    gmsh.model.mesh.field.setAsBackgroundMesh(2)

    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    gmsh.option.setNumber("Mesh.Binary", 1)

    # add physical group
    gmsh.model.setPhysicalName(2, gmsh.model.addPhysicalGroup(2, [fluid_1]), "fluid_1")
    gmsh.model.setPhysicalName(2, gmsh.model.addPhysicalGroup(2, [fluid_2]), "fluid_2")
    gmsh.model.setPhysicalName(1, gmsh.model.addPhysicalGroup(1, e_marker), "interface")
    gmsh.model.setPhysicalName(1, gmsh.model.addPhysicalGroup(1, [e_dry_l, e_dry_r]), "dry")
    gmsh.model.setPhysicalName(1, gmsh.model.addPhysicalGroup(1, [e_wet]), "wet")
    gmsh.model.setPhysicalName(1, gmsh.model.addPhysicalGroup(1, [e_right]), "right")
    gmsh.model.setPhysicalName(1, gmsh.model.addPhysicalGroup(1, [e_top]), "top")
    gmsh.model.setPhysicalName(1, gmsh.model.addPhysicalGroup(1, [e_left]), "left")
    gmsh.model.setPhysicalName(0, gmsh.model.addPhysicalGroup(0, [pts_marker[0], pts_marker[-1]]), "cl")
    gmsh.model.setPhysicalName(0, gmsh.model.addPhysicalGroup(0, [pts_ll, pts_lr]), "clamp")
    # add periodicity
    translation = [1, 0, 0, bbox[1,0] - bbox[0,0], 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    gmsh.model.mesh.setPeriodic(1, [e_right], [e_left], translation)

    # generate and save the mesh
    gmsh.model.mesh.generate(dim=2)
    gmsh.write("mesh/two-phase.msh")
    gmsh.finalize()

def build_unit_square(h: float) -> None:
    gmsh.initialize()
    gmsh.model.add("unit_square")
    # write the points
    p_id = [gmsh.model.geo.addPoint(0.0, 0.0, 0.0, h), 
            gmsh.model.geo.addPoint(1.0, 0.0, 0.0, h), 
            gmsh.model.geo.addPoint(1.0, 1.0, 0.0, h), 
            gmsh.model.geo.addPoint(0.0, 1.0, 0.0, h)]
    # write the edges
    e_id = [gmsh.model.geo.addLine(p_id[0], p_id[1]), 
            gmsh.model.geo.addLine(p_id[1], p_id[2]), 
            gmsh.model.geo.addLine(p_id[2], p_id[3]), 
            gmsh.model.geo.addLine(p_id[3], p_id[0])]
    # add the loop and the surface
    s_1 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(e_id)])
    gmsh.model.geo.synchronize()
    # add physical group
    gmsh.model.setPhysicalName(2, gmsh.model.addPhysicalGroup(2, [s_1]), "domain")
    gmsh.model.setPhysicalName(1, gmsh.model.addPhysicalGroup(1, [e_id[0]]), "bottom")
    gmsh.model.setPhysicalName(1, gmsh.model.addPhysicalGroup(1, [e_id[1]]), "right")
    gmsh.model.setPhysicalName(1, gmsh.model.addPhysicalGroup(1, [e_id[2]]), "top")
    gmsh.model.setPhysicalName(1, gmsh.model.addPhysicalGroup(1, [e_id[3]]), "left")
    # add periodicity
    translation = (1.0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
    gmsh.model.mesh.setPeriodic(1, (e_id[1],), (e_id[3],), translation)
    # translation = (1.0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1)
    # gmsh.model.mesh.setPeriodic(1, (e_id[2],), (e_id[0],), translation)
    
    gmsh.option.setNumber("Mesh.Binary", 1)

    # generate and save
    gmsh.model.mesh.generate(dim = 2)
    gmsh.write("mesh/unit_square.msh")
    gmsh.finalize()

if __name__ == "__main__":
    mesh_name = argv[1] if len(argv) >= 2 else "unit_square"
    if mesh_name == "two-phase":
        bbox = np.array([[-1,0], [1,1]], dtype=np.float64)
        markers = np.array([[0.5,0], [0.5, 0.25], [-0.5, 0.25], [-0.5,0]])
        build_two_phase_mesh(bbox, markers, [0.04, 0.1], 0.5)
    elif mesh_name == "unit_square":
        build_unit_square(0.1)
