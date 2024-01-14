import numpy as np
import gmsh

h_size = 0.05
r = 1.0/4
dryNe = int(np.ceil((0.5 - r) / h_size))
wetNe = int(np.ceil(2*r / h_size))
arcNe = int(np.ceil(np.pi * r / h_size))
#print(f'dryNe = {dryNe}, wetNe = {wetNe}, arcNe = {arcNe}')
# generate the segments
dryPart = np.hstack((np.linspace(0, 1/2-r, dryNe+1)[:,np.newaxis], np.zeros((dryNe+1, 1))))
wetPart = np.hstack((np.linspace(1/2-r, 1/2+r, wetNe+1)[:,np.newaxis], np.zeros((wetNe+1, 1))))
theta = np.linspace(0, np.pi, arcNe+1)[:, np.newaxis]
arc = r * np.hstack((np.cos(theta), np.sin(theta))) + np.array([1/2, 0])

gmsh.initialize()

gmsh.model.add('two-phase')
# add points
dryLeftEnd = 0
for i in range(0, dryNe+1):
    dryLeftEnd = gmsh.model.geo.addPoint(dryPart[i,0], dryPart[i,1], h_size)
wetEnd = 0
for i in range(1, wetNe):
    wetEnd = gmsh.model.geo.addPoint(wetPart[i,0], wetPart[i,1], h_size)
dryRightEnd = 0
for i in range(dryNe,-1,-1):
    dryRightEnd = gmsh.model.geo.addPoint(1 - dryPart[i,0], dryPart[i,1], h_size)
pUpRight = gmsh.model.geo.addPoint(1, 1/2, h_size)
pUpLeft = gmsh.model.geo.addPoint(0, 1/2, h_size)
pArcEnd = 0
for i in range(1, arcNe):
    pArcEnd = gmsh.model.geo.addPoint(arc[i,0], arc[i,1], h_size)

# add boundary lines
for i in range(0, dryNe):
    gmsh.model.geo.addLine(i+1, i+2) # add the left dry part
for i in range(0, wetNe):
    gmsh.model.geo.addLine(dryLeftEnd+i, dryLeftEnd+i+1) # add the wet part
for i in range(0, dryNe):
    gmsh.model.geo.addLine(wetEnd+i+1, wetEnd+i+2) # add the right dry part
gmsh.model.geo.addLine(dryRightEnd, pUpRight)
gmsh.model.geo.addLine(pUpRight, pUpLeft)
lastRectLine = gmsh.model.geo.addLine(pUpLeft, 1)
# add the arc segments
gmsh.model.geo.addLine(dryLeftEnd, pArcEnd)
for i in range(0, arcNe-2):
    gmsh.model.geo.addLine(pArcEnd-i, pArcEnd-i-1)
gmsh.model.geo.addLine(pUpLeft+1, wetEnd+1)
# add curve loop
gmsh.model.geo.addCurveLoop(range(1, lastRectLine+1), 1)
# add plane surface
gmsh.model.geo.addPlaneSurface([1], 1)

gmsh.model.geo.synchronize()

# embed the interface in the bulk mesh
gmsh.model.mesh.embed(1, range(lastRectLine+1, lastRectLine+arcNe+1), 2, 1)
# embed the lower boundary in the bulk mesh
gmsh.model.mesh.embed(1, range(1, 2*dryNe+wetNe+1), 2, 1)

# add physical group
pid1 = gmsh.model.geo.addPhysicalGroup(1, range(1, lastRectLine+1))
print('Physical ID of boundary = {}'.format(pid1))
pid2 = gmsh.model.geo.addPhysicalGroup(1, range(lastRectLine+1, lastRectLine+arcNe+1))
print('Physical ID of interface = {}'.format(pid2))

# generate mesh
gmsh.model.mesh.generate(2)

# output the mesh
gmsh.write('rect.msh')

gmsh.finalize()

