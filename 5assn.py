import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def normalize_mesh(m):
    bbox = m.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    m.translate(-center)
    scale = 1.0 / np.max(bbox.get_extent())
    m.scale(scale, center=np.zeros(3))
    return m

def make_vertical_plane(width=5, height=5):
    w = width / 2
    h = height / 2
    vertices = np.array([
        [0, -h, -w],
        [0,  h, -w],
        [0,  h,  w],
        [0, -h,  w]
    ])
    triangles = np.array([
        [0, 1, 2],
        [0, 2, 3]
    ])
    plane = o3d.geometry.TriangleMesh()
    plane.vertices = o3d.utility.Vector3dVector(vertices)
    plane.triangles = o3d.utility.Vector3iVector(triangles)
    plane.compute_vertex_normals()
    return plane

mesh = o3d.io.read_triangle_mesh("ufo.obj")
mesh.compute_vertex_normals()
mesh = normalize_mesh(mesh)

print("\n=== STEP 1: Original Mesh ===")
print("Vertices:", len(mesh.vertices))
print("Triangles:", len(mesh.triangles))
print("Has vertex colors:", mesh.has_vertex_colors())
print("Has vertex normals:", mesh.has_vertex_normals())

o3d.visualization.draw_geometries([mesh], window_name="STEP 1")

pcd = mesh.sample_points_uniformly(number_of_points=40000)

print("\n=== STEP 2: Point Cloud ===")
print("Points:", len(pcd.points))
print("Has colors:", pcd.has_colors())
print("Has normals:", pcd.has_normals())

o3d.visualization.draw_geometries([pcd], window_name="STEP 2")

print("\n=== STEP 3: Poisson Surface Reconstruction ===")

pcd.estimate_normals()
mesh_rec, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
bbox = pcd.get_axis_aligned_bounding_box()
mesh_crop = mesh_rec.crop(bbox)

print("Vertices:", len(mesh_crop.vertices))
print("Triangles:", len(mesh_crop.triangles))
print("Has vertex colors:", mesh_crop.has_vertex_colors())

normals = np.asarray(mesh_crop.vertex_normals)
normals_norm = (normals + 1) / 2.0
mesh_crop.vertex_colors = o3d.utility.Vector3dVector(normals_norm)

o3d.visualization.draw_geometries([mesh_crop], window_name="STEP 3")

print("\n=== STEP 4: Voxelization ===")

voxel_size = 0.03
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)
voxels = voxel_grid.get_voxels()

print("Voxel size:", voxel_size)
print("Number of voxels:", len(voxels))
print("Has colors: False")
print("Triangles: 0")

o3d.visualization.draw_geometries([voxel_grid], window_name="STEP 4")

print("\n=== STEP 5: Adding Vertical Plane ===")

plane = make_vertical_plane(width=6, height=6)
plane.paint_uniform_color([0.3, 0.3, 0.3])
plane.translate([-0.1, 0, 0])

o3d.visualization.draw_geometries([mesh, plane], window_name="STEP 5")

print("\n=== STEP 6: Vertical Clipping ===")

points = np.asarray(pcd.points)
normal = np.array([1, 0, 0])
p0 = np.array([-0.1, 0, 0])
mask = (points - p0) @ normal > 0

pcd_clipped = o3d.geometry.PointCloud()
pcd_clipped.points = o3d.utility.Vector3dVector(points[mask])

print("Remaining points:", len(pcd_clipped.points))
print("Has colors:", pcd_clipped.has_colors())
print("Has normals:", pcd_clipped.has_normals())
print("Triangles: 0")

o3d.visualization.draw_geometries([pcd_clipped], window_name="STEP 6")

print("\n=== STEP 7: Gradient Coloring & Extremes ===")

points = np.asarray(pcd_clipped.points)
z_vals = points[:, 2]

z_min, z_max = z_vals.min(), z_vals.max()
colors = plt.cm.coolwarm((z_vals - z_min) / (z_max - z_min))[:, :3]
pcd_clipped.colors = o3d.utility.Vector3dVector(colors)

min_idx = np.argmin(z_vals)
max_idx = np.argmax(z_vals)

p_min = points[min_idx]
p_max = points[max_idx]

print("Z-min:", p_min)
print("Z-max:", p_max)

cube_min = o3d.geometry.TriangleMesh.create_box(0.04, 0.04, 0.04)
cube_max = o3d.geometry.TriangleMesh.create_box(0.04, 0.04, 0.04)
cube_min.translate(p_min)
cube_max.translate(p_max)
cube_min.paint_uniform_color([0, 0, 1])
cube_max.paint_uniform_color([1, 0, 0])

o3d.visualization.draw_geometries(
    [pcd_clipped, cube_min, cube_max],
    window_name="STEP 7"
)
