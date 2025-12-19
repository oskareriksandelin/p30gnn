
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

def MirrorTransformation(moment, B, rel_pos, return_plane=False):
    """
    Generate a random mirror matrix and reflect the spin moments, magnetic fields, and relative positions across the plane.
    The vector direction mirrors without translation.

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
    n = torch.randn(3)
    n = n / n.norm()

    R = torch.eye(3) - 2.0 * torch.outer(n, n)

    reflected_moment = moment @ R.T
    reflected_B = B @ R.T
    reflected_rel_pos = rel_pos @ R.T
    if return_plane:
        point = torch.zeros(3)
        normal = n
        return reflected_moment, reflected_B, reflected_rel_pos, point, normal
    else:
        return reflected_moment, reflected_B, reflected_rel_pos

def PermutationTransform():
    """
    Apply a random permutation to the nodes in the graph.


    return permuted_moment, B, permuted_pos, permuted_rel_pos
    """
    pass 

def main():
    pass

if __name__ == "__main__":
    """
    Test the MirrorTransformation function. 
    Generate a random vector and reflect it across a random plane.
    Plot the original and reflected vectors along with the plane.
    1. Generate random data
    2. Apply MirrorTransformation
    3. Plot original and transformed data and the mirror plane
    4. Verify correctness visually
    """

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    # Generate random data
    num_nodes = 2
    pos = torch.randn((num_nodes, 3)) + 1.0  # center around [1, 1, 1]
    moment = torch.randn((num_nodes, 3))
    B = torch.randn(3)

    rel_pos = pos - pos.mean(dim=0)  # relative positions
    reflected_moment, reflected_B, _, point, normal = MirrorTransformation(moment, B, rel_pos, return_plane=True)
    # Plot original and transformed data
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(0, 0, 0, B[0], B[1], B[2], color='r', label='Original B', length=0.5)
    ax.quiver(0, 0, 0, reflected_B[0], reflected_B[1], reflected_B[2], color='b', label='Reflected B', length=0.5)
    ax.scatter(pos[:,0], pos[:,1], pos[:,2], color='g', label='Original Positions')
    ax.scatter(reflected_moment[:,0], reflected_moment[:,1], reflected_moment[:,2], color='m', label='Reflected Moments')
    # Plot mirror plane
    d = -point.dot(normal)
    xx, yy = torch.meshgrid(torch.linspace(-2, 2, 10), torch.linspace(-2, 2, 10))
    zz = (-normal[0] * xx - normal[1] * yy - d) / normal[2]
    ax.plot_surface(xx.numpy(), yy.numpy(), zz.numpy(), alpha=0.5, color='y', label='Mirror Plane')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()
    