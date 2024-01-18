import numpy as np
import gmsh

def build_two_phase_mesh(h_size):
    # define the parameters
    r = 1.0/4
    dryNe = int(np.ceil((0.5 - r) / h_size))
    wetNe = int(np.ceil(2*r / h_size))
    intNe = int(np.ceil(np.pi * r / h_size))
    print(f'dryNe = {dryNe}, wetNe = {wetNe}, intNe = {intNe}')

    gmsh.initialize()
    gmsh.model.add("two-phase")
    # write the points
    ptsDryL = [gmsh.model.geo.addPoint(x, 0, 0, h_size) for x in np.linspace(0, 1/2-r, dryNe+1)]
    ptsWet  = [gmsh.model.geo.addPoint(x, 0, 0, h_size) for x in np.linspace(1/2-r+h_size, 1/2+r-h_size, wetNe-1)]
    ptsDryR = [gmsh.model.geo.addPoint(x, 0, 0, h_size) for x in np.linspace(1/2+r, 1, dryNe+1)]
    ptsUR = gmsh.model.geo.addPoint(1, 1/2, 0, h_size)
    ptsUL = gmsh.model.geo.addPoint(0, 1/2, 0, h_size)
    # add the interface markers
    theta = np.linspace(0, np.pi, intNe+1)
    ptsInt = [gmsh.model.geo.addPoint(r*np.cos(t) + 1/2, r*np.sin(t), 0.0, h_size) for t in theta[1:-1]]

    # add the line segments
    edgeDryL = [gmsh.model.geo.addLine(ptsDryL[i], ptsDryL[i+1]) for i in range(dryNe)]
    edgeWet = [gmsh.model.geo.addLine(ptsDryL[-1], ptsWet[0])]
    edgeWet += [gmsh.model.geo.addLine(ptsWet[i], ptsWet[i+1]) for i in range(wetNe-2)]
    edgeWet += [gmsh.model.geo.addLine(ptsWet[-1], ptsDryR[0])]
    edgeDryR = [gmsh.model.geo.addLine(ptsDryR[i], ptsDryR[i+1]) for i in range(dryNe)]
    edgeRight = gmsh.model.geo.addLine(ptsDryR[-1], ptsUR)
    edgeTop = gmsh.model.geo.addLine(ptsUR, ptsUL)
    edgeLeft = gmsh.model.geo.addLine(ptsUL, ptsDryL[0])
    edgeInt = [gmsh.model.geo.addLine(ptsDryR[0], ptsInt[0])]
    edgeInt += [gmsh.model.geo.addLine(ptsInt[i], ptsInt[i+1]) for i in range(intNe-2)]
    edgeInt += [gmsh.model.geo.addLine(ptsInt[-1], ptsDryL[-1])] # the interface is traversed ccw
    # add the line loop
    loop1 = gmsh.model.geo.addCurveLoop(edgeWet + edgeInt)
    fluid1 = gmsh.model.geo.addPlaneSurface([loop1])
    loop2 = gmsh.model.geo.addCurveLoop(edgeDryL + [-e for e in reversed(edgeInt)] + edgeDryR + [edgeRight, edgeTop, edgeLeft])
    fluid2 = gmsh.model.geo.addPlaneSurface([loop2])
    gmsh.model.geo.synchronize()
    # add physical group
    gmsh.model.setPhysicalName(2, gmsh.model.addPhysicalGroup(2, [fluid1]), "fluid1")
    gmsh.model.setPhysicalName(2, gmsh.model.addPhysicalGroup(2, [fluid2]), "fluid2")
    gmsh.model.setPhysicalName(1, gmsh.model.addPhysicalGroup(1, edgeInt), "interface")
    gmsh.model.setPhysicalName(1, gmsh.model.addPhysicalGroup(1, edgeWet), "wet")
    gmsh.model.setPhysicalName(1, gmsh.model.addPhysicalGroup(1, edgeDryL), "dryLeft")
    gmsh.model.setPhysicalName(1, gmsh.model.addPhysicalGroup(1, edgeDryR), "dryRight")
    gmsh.model.setPhysicalName(1, gmsh.model.addPhysicalGroup(1, [edgeRight]), "right")
    gmsh.model.setPhysicalName(1, gmsh.model.addPhysicalGroup(1, [edgeTop]), "top")
    gmsh.model.setPhysicalName(1, gmsh.model.addPhysicalGroup(1, [edgeLeft]), "left")
    # add periodicity
    translation = [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    gmsh.model.mesh.setPeriodic(1, [edgeRight], [edgeLeft], translation)
    # generate and save the mesh
    gmsh.model.mesh.generate(dim=2)
    gmsh.write("mesh/two-phase.msh")
    gmsh.finalize()

build_two_phase_mesh(0.05)
