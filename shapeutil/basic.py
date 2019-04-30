import math
import numpy as np


PI = math.pi


def arc(x=0, y=0, r=1, theta1=0, theta2=PI, resolution=180):
    #TODO: revisit docstrings
    """
    Returns x and y coords (row 0 and row 1, respectively)
    of arc. "resolution" = number of points per 2*PI rads.
    Input arguments x and y refer to centerpoint of arc.
    """

    thetas = np.linspace(theta1, theta2,
        int(abs(theta2 - theta1) * (resolution / (2*PI))))
    return np.vstack((x + r*np.cos(thetas), y + r*np.sin(thetas)))


def zigzag(start, end, nodes, width):
    """!
    Return a list of points corresponding to a zigzag.

    @param r1 (array-like) The (x, y) coordinates of the first endpoint.
    @param r2 (array-like) The (x, y) coordinates of the second endpoint.
    @param nodes (int) The number of zigzag "nodes" or coils.
    @param width (int or float) The diameter of the zigzag.
    @return An array of x coordinates and an array of y coordinates.
    """

    # Check that nodes is at least 1.
    nodes = max(int(nodes), 1)

    # Convert to numpy array to account for inputs of different types/shapes.
    start, end = np.array(start).reshape((2,)), np.array(end).reshape((2,))

    # If both points are coincident, return the x and y coords of one of them.
    if (start == end).all():
        return start[0], start[1]

    # Calculate length of zigzag (distance between endpoints).
    length = np.linalg.norm(np.subtract(end, start))

    # Calculate unit vectors tangent (u_t) and normal (u_t) to zigzag.
    u_t = np.subtract(end, start) / length
    u_n = np.array([[0, -1], [1, 0]]).dot(u_t)

    # Initialize array of x (row 0) and y (row 1) coords of the nodes+2 points.
    zigzag_coords = np.zeros((2, nodes + 2))
    zigzag_coords[:,0], zigzag_coords[:,-1] = start, end

    # Check that length is not greater than the total length the zigzag
    # can extend (otherwise, math domain error will result), and compute the
    # normal distance from the centerline of the zigzag.
    normal_dist = math.sqrt(max(0, width**2 - (length**2 / nodes**2))) / 2

    # Compute the coordinates of each point (each node).
    for i in range(1, nodes + 1):
        zigzag_coords[:,i] = (
            start
            + ((length * (2 * i - 1) * u_t) / (2 * nodes))
            + (normal_dist * (-1)**i * u_n))

    return zigzag_coords[0,:], zigzag_coords[1,:]
