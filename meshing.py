from mesh_vox import read_and_reshape_stl, voxelize
import pyvista as pv
import numpy as np
# path to the stl file
input_path = 'rail.stl'
# number of voxels used to represent the largest dimension of the 3D model
resolution = 400

# read and rescale
mesh, bounding_box = read_and_reshape_stl(input_path, resolution)
# create voxel array
voxels, bounding_box = voxelize(mesh, bounding_box)

print(voxels.shape)
voxArr = voxels.astype(int)
np.save("Rail.npy",voxArr)
