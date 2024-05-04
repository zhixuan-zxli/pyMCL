from sys import argv
import numpy as np
import gmsh

def build_two_phase_mesh(bbox: np.ndarray, markers: np.ndarray, field_params: np.ndarray) -> None:
    """
    Build the two-phase mesh, given the interface markers. 
    bbox    [in, 2 x 2]  [[x_lo, y_lo], [x_hi, y_hi]]
    markers [in, Np x 2] with the orientation where the outward normal is pointing to fluid 2. 
    field_params  [in] (2,3) [0], [1] for (size_min, size_max, dist_min, dist_max) for the f-f interface and the contact line, respecitvely. 
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
    gmsh.model.mesh.field.setNumber(2, "SizeMin", field_params[0,0])
    gmsh.model.mesh.field.setNumber(2, "SizeMax", field_params[0,1])
    gmsh.model.mesh.field.setNumber(2, "DistMin", field_params[0,2])
    gmsh.model.mesh.field.setNumber(2, "DistMax", field_params[0,3])

    gmsh.model.mesh.field.add("Distance", 3)
    gmsh.model.mesh.field.setNumbers(3, "PointsList", [pts_marker[0], pts_marker[-1]])

    gmsh.model.mesh.field.add("Threshold", 4)
    gmsh.model.mesh.field.setNumber(4, "InField", 3)
    gmsh.model.mesh.field.setNumber(4, "SizeMin", field_params[1,0])
    gmsh.model.mesh.field.setNumber(4, "SizeMax", field_params[1,1])
    gmsh.model.mesh.field.setNumber(4, "DistMin", field_params[1,2])
    gmsh.model.mesh.field.setNumber(4, "DistMax", field_params[1,3])

    gmsh.model.mesh.field.add("Min", 5)
    gmsh.model.mesh.field.setNumbers(5, "FieldsList", [2, 4])

    gmsh.model.mesh.field.setAsBackgroundMesh(5)

    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    gmsh.option.setNumber("Mesh.Algorithm", 6) # 5 = Delaunay, 6 = Frontal-Delaunay
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
    # # add periodicity
    # translation = [1, 0, 0, bbox[1,0] - bbox[0,0], 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    # gmsh.model.mesh.setPeriodic(1, [e_right], [e_left], translation)

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
        # markers = np.array([[0.5,0], [0.5, 0.25], [-0.5, 0.25], [-0.5,0]])
        theta = np.arange(21) / 20 * np.pi
        markers = np.vstack((0.5*np.cos(theta), 0.5*np.sin(theta)))
        build_two_phase_mesh(bbox, markers.T, np.array(((0.08, 0.2, 0.0, 0.5), (0.01, 0.2, 0.03, 0.2))))
    elif mesh_name == "unit_square":
        build_unit_square(0.1)
