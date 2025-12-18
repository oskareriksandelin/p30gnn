
import torch
import math

def RandomRotationTransform(moment, B, pos, rel_pos):
    """
    Apply a random rotation to the spin moments, magnetic fields, and positions.
    Args:
        moment (torch.Tensor): Spin moments tensor of shape (num_nodes, 3)
        B (torch.Tensor): Magnetic fields tensor of shape (3,)
        pos (torch.Tensor): Positions tensor of shape (num_nodes, 3)
    Returns:
        rotated_moment (torch.Tensor): Rotated spin moments
        rotated_B (torch.Tensor): Rotated magnetic fields
        rotated_pos (torch.Tensor): Rotated positions
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
