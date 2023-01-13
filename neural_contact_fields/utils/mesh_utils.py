import numpy as np
import trimesh
from scipy.spatial import KDTree


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
