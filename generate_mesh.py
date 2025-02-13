from sys import argv
import numpy as np
import gmsh

def build_drop_mesh(bbox: np.ndarray, markers: np.ndarray, field_params: np.ndarray) -> None:
    """
    Build the drop mesh, given the interface markers. 
    bbox    [in, 2 x 2]  [[x_lo, y_lo], [x_hi, y_hi]]
    markers [in, Np x 2] with the orientation where the outward normal is pointing to fluid 2. 
    field_params  [in] (2,3) [0], [1] for (size_min, size_max, dist_min, dist_max) for the f-f interface and the contact line, respecitvely. 
    """
    gmsh.initialize()
    gmsh.model.add("drop")
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

    gmsh.option.setNumber("Mesh.Algorithm", 5) # 5 = Delaunay, 6 = Frontal-Delaunay
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
    gmsh.write("mesh/drop.msh")
    gmsh.finalize()

def build_two_phase(bbox: np.ndarray, field_params: np.ndarray) -> None:
    """
    Build the two-phase mesh, given the interface markers. 
    bbox    [in, 2 x 2]  [[x_lo, y_lo], [x_hi, y_hi]]
    field_params  [in] (4,) (size_min, size_max, dist_min, dist_max) for the interface
    """
    gmsh.initialize()
    gmsh.model.add("two-phase")
    # write the points
    pts_box = [
        gmsh.model.geo.addPoint(bbox[0,0], bbox[0,1], 0), # lower left corner
        gmsh.model.geo.addPoint(bbox[1,0], bbox[0,1], 0), # lower right corner
        gmsh.model.geo.addPoint(bbox[1,0], bbox[1,1], 0), # upper right corner
        gmsh.model.geo.addPoint(bbox[0,0], bbox[1,1], 0), # upper left corner
    ]
    pts_i = [
        gmsh.model.geo.addPoint(0.0, bbox[0,1], 0), 
        gmsh.model.geo.addPoint(0.0, bbox[1,1], 0)
    ]

    # add the line segments
    edges = [
        gmsh.model.geo.addLine(pts_box[0], pts_i[0]),   # 0: wet bottom
        gmsh.model.geo.addLine(pts_i[0], pts_box[1]),   # dry bottom
        gmsh.model.geo.addLine(pts_box[1], pts_box[2]), # right
        gmsh.model.geo.addLine(pts_box[2], pts_i[1]),   # 3: dry top
        gmsh.model.geo.addLine(pts_i[1], pts_box[3]),   # wet top
        gmsh.model.geo.addLine(pts_box[3], pts_box[0]), # left
        gmsh.model.geo.addLine(pts_i[0], pts_i[1]),     # 6: interface
    ]
    # add the line loop
    loop_1 = gmsh.model.geo.addCurveLoop([edges[0], edges[6], edges[4], edges[5]])
    fluid_1 = gmsh.model.geo.addPlaneSurface([loop_1])
    loop_2 = gmsh.model.geo.addCurveLoop([edges[1], edges[2], edges[3], -edges[6]])
    fluid_2 = gmsh.model.geo.addPlaneSurface([loop_2])
    gmsh.model.geo.synchronize()

    # add the mesh size fields
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", [edges[0], edges[1], edges[6]])
    # gmsh.model.mesh.field.setNumber(1, "Sampling", 20)

    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", field_params[0])
    gmsh.model.mesh.field.setNumber(2, "SizeMax", field_params[1])
    gmsh.model.mesh.field.setNumber(2, "DistMin", field_params[2])
    gmsh.model.mesh.field.setNumber(2, "DistMax", field_params[3])

    gmsh.model.mesh.field.setAsBackgroundMesh(2)

    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    gmsh.option.setNumber("Mesh.Algorithm", 6) # 5 = Delaunay, 6 = Frontal-Delaunay
    gmsh.option.setNumber("Mesh.Binary", 1)

    # add physical group
    gmsh.model.setPhysicalName(2, gmsh.model.addPhysicalGroup(2, [fluid_1]), "fluid_1")
    gmsh.model.setPhysicalName(2, gmsh.model.addPhysicalGroup(2, [fluid_2]), "fluid_2")
    gmsh.model.setPhysicalName(1, gmsh.model.addPhysicalGroup(1, [edges[6]]), "interface")
    gmsh.model.setPhysicalName(1, gmsh.model.addPhysicalGroup(1, [edges[0]]), "wet_sheet")
    gmsh.model.setPhysicalName(1, gmsh.model.addPhysicalGroup(1, [edges[1]]), "dry_sheet")
    gmsh.model.setPhysicalName(1, gmsh.model.addPhysicalGroup(1, [edges[4]]), "wet_top")
    gmsh.model.setPhysicalName(1, gmsh.model.addPhysicalGroup(1, [edges[3]]), "dry_top")
    gmsh.model.setPhysicalName(1, gmsh.model.addPhysicalGroup(1, [edges[5]]), "left")
    gmsh.model.setPhysicalName(1, gmsh.model.addPhysicalGroup(1, [edges[2]]), "right")
    gmsh.model.setPhysicalName(0, gmsh.model.addPhysicalGroup(0, [pts_i[0]]), "cl")
    gmsh.model.setPhysicalName(0, gmsh.model.addPhysicalGroup(0, [pts_i[1]]), "cl_top")
    gmsh.model.setPhysicalName(0, gmsh.model.addPhysicalGroup(0, [pts_box[0], pts_box[1]]), "clamp")
    # add periodicity
    # translation = [1, 0, 0, bbox[1,0] - bbox[0,0], 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    # gmsh.model.mesh.setPeriodic(1, [edges[2]], [edges[5]], translation)

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

def build_dumbbell_mesh(radius: float, distance: float, mesh_size: float) -> None:
    """
    Build a dumbbell surface mesh in 3D using Gmsh.
    
    Parameters:
    radius (float): Radius of the spheres.
    distance (float): Distance between the centers of the two spheres.
    mesh_size (float): Mesh size for the surface mesh.
    """
    gmsh.initialize()
    gmsh.model.add("dumbbell")

    # Define the centers of the two spheres
    center1 = [-distance/2, 0, 0]
    center2 = [distance/2, 0, 0]

    # Add the points for the spheres
    sphere1 = gmsh.model.occ.addSphere(center1[0], center1[1], center1[2], radius)
    sphere2 = gmsh.model.occ.addSphere(center2[0], center2[1], center2[2], radius)

    # Fuse the two spheres to create the dumbbell shape
    gmsh.model.occ.fuse([(3, sphere1)], [(3, sphere2)])

    # Synchronize the CAD kernel with the Gmsh model
    gmsh.model.occ.synchronize()

    # Set mesh size
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)

    # Generate the mesh
    gmsh.model.mesh.generate(2)

    # Tag all 2D triangles with number 2
    gmsh.model.addPhysicalGroup(2, [s[1] for s in gmsh.model.getEntities(2)], 2)

    # Save the mesh to a binary file
    gmsh.option.setNumber("Mesh.Binary", 1)
    gmsh.write("mesh/dumbbell.msh")

    # Finalize Gmsh
    gmsh.finalize()

def build_modulated_circle_mesh(amplitude: float, num_periods: int, num_points: int) -> None:
    """
    Build a 1D mesh in 2D for a unit circle modulated in the normal direction by a sinusoidal wave.
    
    Parameters:
    amplitude (float): Amplitude of the sinusoidal modulation.
    num_points (int): Number of points to discretize the circle.
    """
    gmsh.initialize()
    gmsh.model.add("modulated_circle")

    # Define the points for the modulated circle
    theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x = np.cos(theta)
    y = np.sin(theta)
    modulation = amplitude * np.sin(num_periods * theta)
    x_modulated = x + modulation * np.cos(theta)
    y_modulated = y + modulation * np.sin(theta)

    # Add the points to Gmsh
    points = []
    for i in range(num_points):
        points.append(gmsh.model.geo.addPoint(x_modulated[i], y_modulated[i], 0))

    # Add the lines to form the modulated circle
    lines = []
    for i in range(num_points):
        lines.append(gmsh.model.geo.addLine(points[i], points[(i + 1) % num_points]))

    # Create a line loop and a plane surface
    line_loop = gmsh.model.geo.addCurveLoop(lines)
    gmsh.model.geo.addPlaneSurface([line_loop])

    # Synchronize the CAD kernel with the Gmsh model
    gmsh.model.geo.synchronize()

    # Generate the mesh
    gmsh.model.mesh.generate(1)

    # Save the mesh to a binary file
    gmsh.option.setNumber("Mesh.Binary", 1)
    gmsh.write("mesh/modulated_circle.msh")

    # Finalize Gmsh
    gmsh.finalize()


if __name__ == "__main__":
    if len(argv) < 2:
        print("python3 generate_mesh.py mesh_name")
        quit()
    if argv[1] == "dumbbell":
        build_dumbbell_mesh(radius=1.0, distance=1.6, mesh_size=0.2)
    elif argv[1] == "modulated_circle":
        build_modulated_circle_mesh(amplitude=0.1, num_periods=10, num_points=129)
    elif argv[1] == "drop":
        bbox = np.array([[-1,0], [1,1]], dtype=np.float64)
        theta_0 = np.pi*2/3 # change this
        print("Theta_0 = {}".format(theta_0*180/np.pi))
        vol_0 = np.pi/8 # the volume of the droplet
        h_min, h_max = 0.03, 0.2
        R = np.sqrt(vol_0 / (theta_0 - 0.5 * np.sin(2*theta_0)))
        arcl = 2*R*theta_0
        num_segs = np.ceil(arcl / h_min)
        theta = np.arange(num_segs+1) / num_segs * (2*theta_0) + np.pi/2 - theta_0 # the theta span
        markers = np.vstack((R * np.cos(theta), R * (np.sin(theta) - np.cos(theta_0))))
        markers[1,0] = 0.0; markers[1,-1] = 0.0 # attach on the substrate
        build_drop_mesh(bbox, markers.T, np.array(((h_min, h_max, 0.0, 0.5), (1e-4, 0.2, 1e-3, 0.2))))
    elif argv[1] == "two-phase":
        bbox=np.array(((-1.0, 0.0), (1.0, 0.75)))
        build_two_phase(bbox, (0.06, 0.25, 0.0, 0.5))
    elif argv[1] == "unit_square":
        build_unit_square(0.1)
    else:
        print("Unknown mesh name")
        quit()
