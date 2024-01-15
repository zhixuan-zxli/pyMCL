import numpy as np
import meshio
import gmsh
import pygmsh

def build_two_phase_mesh(h_size):
  # define the parameters
  r = 1.0/4
  dryNe = int(np.ceil((0.5 - r) / h_size))
  wetNe = int(np.ceil(2*r / h_size))
  intNe = int(np.ceil(np.pi * r / h_size))
  print(f'dryNe = {dryNe}, wetNe = {wetNe}, intNe = {intNe}')
  # start writing the geometry
  with pygmsh.geo.Geometry() as geom:
    # write the points
    ptsDryL = [geom.add_point([x,0], h_size) for x in np.linspace(0, 1/2-r, dryNe+1)]
    ptsWet  = [geom.add_point([x,0], h_size) for x in np.linspace(1/2-r+h_size, 1/2+r-h_size, wetNe-1)]
    ptsDryR = [geom.add_point([x,0], h_size) for x in np.linspace(1/2+r, 1, dryNe+1)]
    ptsUR = geom.add_point([1,1/2], h_size)
    ptsUL = geom.add_point([0,1/2], h_size)
    # add the interface markers
    theta = np.linspace(0, np.pi, intNe+1)
    ptsInt = [geom.add_point([r * np.cos(t) + 1/2, r * np.sin(t)], h_size) for t in theta[1:-1]]
    # add the line segments
    edgeDryL = [geom.add_line(ptsDryL[i], ptsDryL[i+1]) for i in range(dryNe)]
    edgeWet = [geom.add_line(ptsDryL[-1], ptsWet[0])]
    edgeWet += [geom.add_line(ptsWet[i], ptsWet[i+1]) for i in range(wetNe-2)]
    edgeWet += [geom.add_line(ptsWet[-1], ptsDryR[0])]
    edgeDryR = [geom.add_line(ptsDryR[i], ptsDryR[i+1]) for i in range(dryNe)]
    edgeRight = geom.add_line(ptsDryR[-1], ptsUR)
    edgeTop = geom.add_line(ptsUR, ptsUL)
    edgeLeft = geom.add_line(ptsUL, ptsDryL[0])
    edgeInt = [geom.add_line(ptsDryR[0], ptsInt[0])]
    edgeInt += [geom.add_line(ptsInt[i], ptsInt[i+1]) for i in range(intNe-2)]
    edgeInt += [geom.add_line(ptsInt[-1], ptsDryL[-1])] # the interface is traversed ccw
    # add the line loop
    loop1 = geom.add_curve_loop(edgeWet + edgeInt)
    fluid1 = geom.add_plane_surface(loop1)
    #
    geom.synchronize()
    # add physical group
    geom.add_physical([fluid1], "fluid1")
    # generate and save the mesh
    geom.generate_mesh(dim=2)
    pygmsh.write('two-phase-pygmsh.msh')
  # return mesh

build_two_phase_mesh(0.05)
