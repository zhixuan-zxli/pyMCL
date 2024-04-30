# pyMCL

**Compiling a dependence**
```
cd tools
python3 setup.py build_ext --inplace
```

**Generating videos from image sequence**
```
ffmpeg -pattern_type glob -i "*.png" -c:v libx264 -pix_fmt yuv420p out.mp4
```
