
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
    Generate a random parity transformation on of of the axes. 
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
    R = torch.eye(3)
    axis = torch.randint(0, 3, (1,)).item()
    R[axis, axis] = -1  # Reflect across the chosen axis

    reflected_moment = moment @ R.T
    reflected_B = B @ R.T
    reflected_pos = pos @ R.T
    reflected_rel_pos = rel_pos @ R.T
    
    #return reflected_moment, reflected_B, reflected_rel_pos    # to plot the relection plane
    return reflected_moment, reflected_B, reflected_pos, reflected_rel_pos
    

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
    num_nodes = 10
    pos = torch.randn((num_nodes, 3)) + 1.0  # center around [1, 1, 1]
    moment = torch.randn((num_nodes, 3))
    B = torch.randn(3)

    reflected_moment, reflected_B, reflected_pos, _ = MirrorTransformation(moment, B, pos, pos)
    res_mom, res_B, res_pos, _ = RandomRotationTransform(reflected_moment, reflected_B, reflected_pos, reflected_pos)

    # Plot original and reflected data
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(0  , 0, 0, moment[0,0], moment[0,1], moment[0,2], color='b', label='Original Moment')
    ax.quiver(0, 0, 0, reflected_moment[0,0], reflected_moment[0,1], reflected_moment[0,2], color='r', label='Reflected Moment')
    ax.quiver(0, 0, 0, res_mom[0,0], res_mom[0,1], res_mom[0,2], color='g', label='Rotated Reflected Moment')
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_zlim([-3, 3])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title('Original, Reflected, and Rotated Reflected Spin Moments')
    plt.show()
