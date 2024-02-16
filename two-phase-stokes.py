import numpy as np
from dolfin import *
from msh2xdmf import import_mesh
from util import *
# from matplotlib import pyplot

def testStokes():
    # load the mesh
    bulk_mesh, boundary_marker, subdomain_marker, assoc_table = \
        import_mesh(prefix='two-phase', subdomains=True, tdim=2, gdim=2, directory='mesh')
    interface_mesh = MeshView.create(boundary_marker, assoc_table["interface"])
    interface_mesh.init_cell_orientations(Expression(("x[0]", "x[1]"), degree = 1))

    # define the symbols
    dx = Measure('dx', domain=bulk_mesh, subdomain_data=subdomain_marker)
    ds = Measure('ds', domain=bulk_mesh, subdomain_data=boundary_marker)
    dS = Measure('dx', domain=interface_mesh)
    dp = Measure('ds', domain=interface_mesh)
    n = CellNormal(interface_mesh)
    # just to check orientation
    proj_mea = dot(Constant((0.0, 1.0)), n) * dS
    print('Projected measure of interface = {}'.format(assemble(proj_mea)))

    # Define the periodic boundary condition
    class PeriodicBoundary(SubDomain):
        # identify the left boundary
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], -1.0)     # change this if the domain changes <<<<<<<<<<<<<
        # map the right boundary (x) to the left (y)
        def map(self, x, y):
            y[0] = x[0] - 2.0
            y[1] = x[1]
    periodic_bc = PeriodicBoundary()

    # define the function spaces
    FuncSpaces = dict()
    FuncSpaces["U"] = VectorFunctionSpace(bulk_mesh, 'CG', 2, constrained_domain = periodic_bc)
    FuncSpaces["P1"] = FunctionSpace(bulk_mesh, 'CG', 1, constrained_domain = periodic_bc)
    FuncSpaces["P0"] = FunctionSpace(bulk_mesh, 'DG', 0, constrained_domain = periodic_bc)
    FuncSpaces["X"] = VectorFunctionSpace(interface_mesh, 'CG', 1)
    FuncSpaces["K"] = FunctionSpace(interface_mesh, 'CG', 1)

    FuncSpace = MixedFunctionSpace(*FuncSpaces.values())    
    FuncSpaceDims = np.array([0] + [v.dim() for v in FuncSpaces.values()])
    FuncSpaceDims = np.cumsum(FuncSpaceDims)

    X_m = Function(FuncSpaces['X']) # the parametrization of the interface

    # convert the mesh function to a DG0 function for output
    subdomain_fn = cellFuncToDG0(bulk_mesh, FuncSpaces["P0"], subdomain_marker)
    
    # for moving the bulk mesh
    W_space = FunctionSpace(bulk_mesh, bulk_mesh.ufl_coordinate_element()) # should be 2D P-1 element
    bulk_disp = Function(W_space)
    bulk_X = Function(W_space)
    
    # define the elastic bilinear form
    w = TrialFunction(W_space)
    s = TestFunction(W_space)
    modulus = Function(FuncSpaces["P0"])
    a_e = modulus * inner(Constant(0.5)*(grad(w) + grad(w).T) + div(w)*Identity(2), Constant(0.5)*(grad(s) + grad(s).T)) * dx
    l_e = dot(Constant((0.0, 0.0)), s) * dx
    
    # build the vertex dof-to-dof map
    v_map = np.array(interface_mesh.topology().mapping()[bulk_mesh.id()].vertex_map(), dtype=np.uint64)
    d2v = dof_to_vertex_map(FuncSpaces["X"])
    v2d = vertex_to_dof_map(W_space)
    d2d = v2d[v_map[d2v//2]*2 + (d2v%2)]

    # define the flow boundary conditions
    noSlipBC = DirichletBC(FuncSpace.sub_space(0), Constant((0.0, 0.0)), boundary_marker, assoc_table["top"])
    noPenBC = DirichletBC(FuncSpace.sub_space(0).sub(1), Constant(0.0), boundary_marker, assoc_table["bottom"])

    # Define the interface boundary condition
    def onMeshBoundary(x, on_boundary):
        return on_boundary
    attachBC = DirichletBC(FuncSpace.sub_space(3).sub(1), Constant(0.0), onMeshBoundary)
    # conormal_1_x = Expression("x[0] > 0 ? 1.0 : -1.0", degree=1) # the x component of the conormal of the wet part
    
    essentialBCs = [noSlipBC, noPenBC, attachBC]

    # define the physical parameters
    phys = {"Re":10.0, "Ca":0.1, "ls":0.1, "beta":0.1, "beta_c": 0.1, "theta_Y":2*np.pi/3, "cl_speed":5.0}
    params = {"dt":1e-4, "max_step":200}

    # define the variational problem
    (u, p1, p0, X, kappa) = TrialFunctions(FuncSpace)
    (v, q1, q0, Y, eta) = TestFunctions(FuncSpace)
    a_fl = (Constant(1.0/phys["Re"]) * inner(grad(u), grad(v)) - div(v)*(p1+p0) + div(u)*(q1+q0)) * dx  # incompressible fluid
    a_sl = Constant(phys["beta"]/phys["ls"]) * u[0] * v[0] * ds(assoc_table["bottom"])                  # slip boundary condition
    a_ca = -Constant(1.0/phys["Ca"]) * kappa * dot(n, v) * dS                                           # capillary tension
    a_i  = ( Constant(1.0/params["dt"]) * dot(X, n)*eta - dot(u, n)*eta \
      + kappa * dot(n, Y) + inner(grad(X), grad(Y)) ) * dS                                             # interface motion
    # a_cl = Constant(phys["beta_c"]*phys["Ca"]/params["dt"]) * X[0] * Y[0] * dp                         # contact line condition
    l_i  = Constant(1.0/params["dt"]) * dot(X_m, n)*eta * dS  # interface motion
    # l_cl = Constant(phys["beta_c"]*phys["Ca"]/params["dt"]) * X_m[0] * Y[0] * dp \
    #   + Constant(np.cos(phys["theta_Y"])) * conormal_1_x * Y[0] * dp    # contact line condition
    
    a = a_fl + a_sl + a_ca + a_i # + a_cl
    l = l_i #  + l_cl

    # create the solution functions
    sol = Function(FuncSpace)
    subSol = [Function(v) for v in FuncSpaces.values()]
    subSol[0].rename("u", "velocity")

    outfile_u = XDMFFile("data/two-phase-u.xdmf")
    outfile_phase = XDMFFile("data/two-phase-phase.xdmf")
    # outfile_disp = XDMFFile("data/two-phase-disp.xdmf")

    t = params["dt"]
    for m in range(params["max_step"]):
        print('t = {0:.4f}'.format(t))

        # synchornize the surface parametrization
        get_coordinates(X_m, interface_mesh.geometry())

        # get the old contact line positions
        cl_m = [assemble(Expression("x[0] <= 0 ? 1.0 : 0.0", degree=1) * X_m[0] * dp),  # left cl
                assemble(Expression("x[0] >= 0 ? 1.0 : 0.0", degree=1) * X_m[0] * dp)]  # right cl

        # set up the prescribed contact line condition
        cl = [-0.5 + phys["cl_speed"] * t, 0.5 - phys["cl_speed"] * t] 
        clExpr = Expression("x[0] >=0 ? cl : (-cl)", cl=cl[1], degree=1)
        clBC = DirichletBC(FuncSpace.sub_space(3).sub(0), clExpr, onMeshBoundary)
        # print("cl_m = {}, cl = {}".format(cl_m, cl))

        # assemble the linear system for fluid and interface
        system = assemble_mixed_system(a == l, sol, essentialBCs + [clBC])
        A_blocks = system[0]
        rhs_blocks = system[1]

        A = PETScNestMatrix(A_blocks)
        L = Vector()
        sol_vec = Vector()
        A.init_vectors(L, rhs_blocks)
        A.init_vectors(sol_vec, [s.vector() for s in subSol])

        # solve the linear system
        A.convert_to_aij()
        solve(A, sol_vec, L) # use LU solver

        # get the sub solution
        for i in range(4):
            subSol[i].vector().set_local(sol_vec.get_local()[FuncSpaceDims[i]:FuncSpaceDims[i+1]])
            subSol[i].vector().apply("")

        # get the new contact line position
        # cl = [assemble(Expression("x[0] <= 0 ? 1.0 : 0.0", degree=1) * subSol[3][0] * dp),  # left cl
        #       assemble(Expression("x[0] >= 0 ? 1.0 : 0.0", degree=1) * subSol[3][0] * dp)]  # right cl
        cl_disp = [cl[0] - cl_m[0], cl[1] - cl_m[1]]

        # embed the interface displacement into the bulk mesh
        vec_disp = subSol[3].vector().get_local() - X_m.vector().get_local()
        vec_bulk_disp = bulk_disp.vector().get_local()
        vec_bulk_disp[:] = 0
        vec_bulk_disp[d2d] += vec_disp
        bulk_disp.vector().set_local(vec_bulk_disp)
        bulk_disp.vector().apply("")

        # before the mesh is displaced, output the solution
        outfile_u.write(subSol[0], t)
        outfile_phase.write(subdomain_fn, t)
                
        # Supply the displacement BC at the bottom
        # In 3D, this should be obtained from mesh adjustment.
        # Here we provide the explicit displacement.
        disp_BC = Expression("(x[0] <= cl_m_l) ? ((x[0]+1.0)/(cl_m_l+1.0)*cl_disp_l) : ((x[0] >= cl_m_r) ? ((1.0-x[0])/(1.0-cl_m_r)*cl_disp_r) : ((x[0]-cl_m_l)/(cl_m_r-cl_m_l)*cl_disp_r + (cl_m_r-x[0])/(cl_m_r-cl_m_l)*cl_disp_l))", degree=1, cl_m_l=cl_m[0], cl_m_r=cl_m[1], cl_disp_l=cl_disp[0], cl_disp_r=cl_disp[1])
        # Change the magic constant if the domain changes ^^^^^^

        # build the BC for the displacement problem
        noMoveBC_top = DirichletBC(W_space, Constant((0.0, 0.0)), boundary_marker, assoc_table["top"])
        noMoveBC_left = DirichletBC(W_space, Constant((0.0, 0.0)), boundary_marker, assoc_table["left"])
        noMoveBC_right = DirichletBC(W_space, Constant((0.0, 0.0)), boundary_marker, assoc_table["right"])
        moveBC_bot_x = DirichletBC(W_space.sub(0), disp_BC, boundary_marker, assoc_table["bottom"])
        moveBC_bot_y = DirichletBC(W_space.sub(1), Constant(0.0), boundary_marker, assoc_table["bottom"])
        moveBC_int = DirichletBC(W_space, bulk_disp, boundary_marker, assoc_table["interface"])
        
        essentialBCs_e = [noMoveBC_top, noMoveBC_left, noMoveBC_right, moveBC_bot_x, moveBC_bot_y, moveBC_int]

        # calculate the modulus based on element size
        cell_vol = project(CellVolume(bulk_mesh), FuncSpaces["P0"]) # get the cell volume in DG0 space
        cell_vol_max = cell_vol.vector().get_local().max()
        cell_vol_min = cell_vol.vector().get_local().min()
        modulus.assign(project(Constant(1.0) + Constant(cell_vol_max-cell_vol_min) / CellVolume(bulk_mesh), FuncSpaces["P0"]))

        # solve the elastic problem for the mesh displacement
        solve(a_e == l_e, bulk_disp, essentialBCs_e)

        # outfile_disp.write(bulk_disp, t)

        # move the bulk mesh
        get_coordinates(bulk_X, bulk_mesh.geometry())
        bulk_X.vector().axpy(1.0, bulk_disp.vector())
        bulk_X.vector().apply("")
        set_coordinates(bulk_mesh.geometry(), bulk_X)

        # move the interface itself
        set_coordinates(interface_mesh.geometry(), subSol[3])

        t = t + params["dt"]
        
    # relex the mesh

    # convert the pressure to DG1
    # DG1_space = FunctionSpace(bulk_mesh, 'DG', 1)
    # p1_proj = project(subSol[1], DG1_space)
    # p0_proj = project(subSol[2], DG1_space)
    # p1_proj.vector().add_local(p0_proj.vector().get_local())
    # p1_proj.vector().apply("")
    # zeroMean(p1_proj, dx)
        
    outfile_u.close()
    # outfile_disp.close()
    outfile_phase.close()


testStokes()
