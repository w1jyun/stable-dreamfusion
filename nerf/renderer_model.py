import trimesh
import numpy as np
import platform
import os
import pyrender
import matplotlib.pyplot as plt

def render(poses, mesh, light, camera):
    scene = pyrender.Scene(ambient_light=[1, 1, 1], bg_color=[1, 0, 0])
    scene.add(mesh, pose= np.eye(4))
    scene.add(light, pose= np.eye(4))
    scene.add(camera, pose= np.eye(4))

    position = poses.cpu().numpy()
    for node in scene.get_nodes():
        if (node.camera != None):
            scene.set_pose(node, pose=np.reshape(position, (4, 4)))
        elif (node.light != None):
            scene.set_pose(node, pose=np.reshape(position, (4, 4)))

    # set_pose
    r = pyrender.OffscreenRenderer(512, 512)
    color, _ = r.render(scene)

    # plt.figure(figsize=(8,8))
    # plt.imshow(color)
    plt.imsave('res.png', color)
    plt.close()
    return color


def renderSkeleton(poses):
    if platform.system() == "Linux":
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
    model = trimesh.load('./pose/1.gltf', force='mesh', file_type='gltf')
    mesh = pyrender.Mesh.from_trimesh(model, smooth=False)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=15000)

    # yfov, znear=0.05, zfar=None, aspectRatio=None, name=None
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, znear=0.1, zfar=10000, aspectRatio=1)
    return render(poses, mesh, light, camera)

# origin Python 3.7.16
