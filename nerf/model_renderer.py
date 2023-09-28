import trimesh
import numpy as np
import platform
import os
import pyrender
import matplotlib.pyplot as plt

class ModelRenderer():
    def __init__(self, path):
        if platform.system() == "Linux":
            os.environ['PYOPENGL_PLATFORM'] = 'egl'
        model = trimesh.load('w_'+path, force='mesh', file_type='glb')
        mesh_w = pyrender.Mesh.from_trimesh(model, smooth=False)
        model_b = trimesh.load('b_'+path, force='mesh', file_type='glb')
        mesh_b = pyrender.Mesh.from_trimesh(model_b, smooth=False)
        camera = pyrender.PerspectiveCamera(yfov=0.5, znear=0.1, zfar=10000, aspectRatio=1)

        scene = pyrender.Scene(ambient_light=(1, 1, 1), bg_color=(0.0, 0.0, 1.0, 1.0))
        scene.add(mesh_w, name='white', pose= np.eye(4))
        scene.add(mesh_b, name='black', pose= np.eye(4))
        scene.add(camera, pose= np.eye(4))
        self.scene = scene
        # yfov, znear=0.05, zfar=None, aspectRatio=None, name=None

    def render(self, poses, fov):
        position = poses.cpu().numpy()
        scene = self.scene
        for node in scene.get_nodes():
            if (node.camera != None):
                node.camera = pyrender.PerspectiveCamera(yfov=fov, znear=0.1, zfar=10000, aspectRatio=1)
                scene.set_pose(node, pose=np.reshape(position, (4, 4)))
            elif (node.mesh != None):
                if node.name == 'white':
                    node.mesh.primitives[0].material.emissiveFactor = [1.0, 1.0, 1.0, 1.0]
                else:
                    node.mesh.primitives[0].material.emissiveFactor = [0.0, 0.0, 0.0, 1.0]

        # set_pose
        r = pyrender.OffscreenRenderer(512, 512)
        color, _ = r.render(scene)
        plt.imsave('res.png', color)
        plt.close()
        return color