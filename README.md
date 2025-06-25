# pyMCL

**Prerequisites**

`numpy`, `scipy`, `matplotlib`, `meshio`. `scikit-umfpack` is recommmended. `gmsh` is needed for mesh generation. 

**Generating videos from image sequence**
```
ffmpeg -pattern_type glob -i "*.png" -framerate 25 -r 25 -c:v libx264 -pix_fmt yuv420p out.mp4
```

**Usage**

First generate the mesh:
```
mkdir mesh
python generateMesh.py half_drop
```

Then run the simulation:
```
python testDrop.py mesh/half_drop.msh result/ --vis --checkpoint 32
```
