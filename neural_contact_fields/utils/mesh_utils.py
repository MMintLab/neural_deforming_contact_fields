import numpy as np
import trimesh
from scipy.spatial import KDTree
import open3d as o3d


def find_in_contact_triangles(tri_mesh: trimesh.Trimesh, contact_points: np.ndarray):
    """
    Given the triangle mesh of the tool and the contact points (which are all vertices of the tool mesh),
    find the corresponding triangles in the mesh.
    """
    vertices = tri_mesh.vertices
    triangles = tri_mesh.faces
    contact_vertices = np.zeros(len(vertices), dtype=bool)

    # Determine the vertices in contact.
    kd_tree = KDTree(vertices)
    _, contact_points_vert_idcs = kd_tree.query(contact_points)
    contact_vertices[contact_points_vert_idcs] = True

    # Determine if each triangle is in contact. Being in contact means ALL vertices of triangle are in contact.
    contact_triangles = np.array(
        [contact_vertices[tri[0]] and contact_vertices[tri[1]] and contact_vertices[tri[2]] for tri in triangles])

    # Find total area of contact patch.
    mesh = trimesh.Trimesh(tri_mesh.vertices, tri_mesh.triangles)
    contact_area = contact_triangles.astype(float) @ mesh.area_faces

    return contact_vertices, contact_triangles, contact_area


# TODO: Unit test this.
def occupancy_check(mesh: trimesh.Trimesh, query_points: np.ndarray):
    vertices = o3d.utility.Vector3dVector(mesh.vertices)
    triangles = o3d.utility.Vector3iVector(mesh.faces)
    mesh_o3d = o3d.geometry.TriangleMesh(vertices, triangles)
    mesh_o3d_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh_o3d)

    # Create a scene and add the triangle mesh
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh_o3d_t)

    query_points = o3d.core.Tensor(query_points, dtype=o3d.core.Dtype.Float32)
    occupancy = scene.compute_occupancy(query_points)
    occupancy = occupancy.numpy() > 0.5

    return occupancy


def sample_surface_points_in_contact(mesh: trimesh.Trimesh, contact_triangles: np.ndarray, n: int = 1000):
    mesh.fix_normals()

    # Weight sampling to only be on triangles in contact, weighted by the area of those triangles.
    face_weight = mesh.area_faces * contact_triangles

    # Sample on the surface.
    contact_points, _ = trimesh.sample.sample_surface(mesh, count=n, face_weight=face_weight)

    return np.array(contact_points)
