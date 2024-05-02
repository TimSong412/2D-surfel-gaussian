import trimesh
import numpy as np
box = trimesh.creation.box(extents=(1.0, 2.0, 3.0))

subdivided = box.subdivide()
subdivided = subdivided.subdivide() 
colors = np.random.randint(0, 255, (subdivided.faces.shape[0], 4), dtype=np.uint8)

subdivided.visual.face_colors = colors

subdivided.export('grey_box.ply')
