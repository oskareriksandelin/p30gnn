
import torch
import math

def RandomRotationTransform(moment, B, pos, rel_pos):
    """
    Apply a random rotation to the spin moments, magnetic fields, and positions.
    Args:
        moment (torch.Tensor): Spin moments tensor of shape (num_nodes, 3)
        B (torch.Tensor): Magnetic fields tensor of shape (3,)
        pos (torch.Tensor): Positions tensor of shape (num_nodes, 3)
        rel_pos (torch.Tensor): 
    Returns:
        rotated_moment (torch.Tensor): Rotated spin moments
        rotated_B (torch.Tensor): Rotated magnetic fields
        rotated_pos (torch.Tensor): Rotated positions
        rotated_rel_po (torch.Tensor): Rotated relative positions
    """

    # Generate random rotation angles
    theta = torch.rand(1).item() * 2 * math.pi  # Rotation around z-axis
    phi = torch.rand(1).item() * 2 * math.pi    # Rotation around y-axis
    psi = torch.rand(1).item() * 2 * math.pi    # Rotation around x-axis

    # Rotation matrices
    R_z = torch.tensor([[math.cos(theta), -math.sin(theta), 0],
                        [math.sin(theta),  math.cos(theta), 0],
                        [0,               0,              1]])

    R_y = torch.tensor([[math.cos(phi), 0, math.sin(phi)],
                        [0,             1, 0],
                        [-math.sin(phi),0, math.cos(phi)]])

    R_x = torch.tensor([[1, 0,              0],
                        [0, math.cos(psi), -math.sin(psi)],
                        [0, math.sin(psi),  math.cos(psi)]])

    # Combined rotation matrix
    R = R_z @ R_y @ R_x

    # Apply rotation
    rotated_moment = moment @ R.T
    rotated_B = B @ R.T
    rotated_pos = pos @ R.T
    rotated_res_pos = rel_pos @ R.T

    return rotated_moment, rotated_B, rotated_pos, rotated_res_pos

def MirrorTransformation(moment, B, pos, rel_pos):
    """
    Generates a random plane and reflect the spin moments, magnetic fields, and positions.
    The cells corners are in [0, 0, 0] to [2, 2, 2], so the plane is generated within this range. The plane should be defined by a random normal vector from
    the center of the cell [1, 1, 1].

    Args:
        moment (torch.Tensor): Spin moments tensor of shape (num_nodes, 3)
        B (torch.Tensor): Magnetic fields tensor of shape (3,)
        pos (torch.Tensor): Positions tensor of shape (num_nodes, 3)
        rel_pos (torch.Tensor): 
    Returns:
        reflected_moment (torch.Tensor): Reflected spin moments
        reflected_B (torch.Tensor): Reflected magnetic fields
        reflected_pos (torch.Tensor): Reflected positions
        reflected_rel_pos (torch.Tensor): Reflected relative positions
    """
    #TODO: Ensure that the plane intersects the cell

    # Generate a random normal vector for the plane
    normal_vector = torch.randn(3)
    normal_vector = normal_vector / torch.norm(normal_vector)
    d = -torch.dot(normal_vector, torch.tensor([1.0, 1.0, 1.0]))  # Plane passes through the center [1, 1, 1]

    # Function to reflect a point across the plane
    def reflect_point(point):
        distance = torch.dot(normal_vector, point) + d
        reflected_point = point - 2 * distance * normal_vector
        return reflected_point
    
    # Reflect positions
    reflected_pos = torch.stack([reflect_point(p) for p in pos])
    reflected_rel_pos = torch.stack([reflect_point(p) for p in rel_pos])
    # Reflect spin moments and magnetic fields
    reflected_moment = moment - 2 * (moment @ normal_vector)[:, None] * normal_vector
    reflected_B = B - 2 * (B @ normal_vector) * normal_vector

    return reflected_moment, reflected_B, reflected_pos, reflected_rel_pos

def PermutationTransform():
    """
    Apply a random permutation to the nodes in the graph.


    return permuted_moment, B, permuted_pos, permuted_rel_pos
    """
    pass 