# pyMCL

**Generating videos from image sequence**
```
ffmpeg -pattern_type glob -i "*.png" -framerate 25 -r 25 -c:v libx264 -pix_fmt yuv420p out.mp4
```
