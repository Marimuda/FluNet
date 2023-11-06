import enum


class NodeType(enum.IntEnum):
    """Enumeration of node types for a graph or network.

    Attributes:
        NORMAL (int): Represents a normal node.
        OBSTACLE (int): Represents an obstacle node.
        AIRFOIL (int): Represents an airfoil node.
        HANDLE (int): Represents a handle node.
        INFLOW (int): Represents an inflow node.
        OUTFLOW (int): Represents an outflow node.
        WALL_BOUNDARY (int): Represents a wall boundary node.
        SIZE (int): Represents the size of the enum for iteration or checks.
    """

    NORMAL = 0
    OBSTACLE = 1
    AIRFOIL = 2
    HANDLE = 3
    INFLOW = 4
    OUTFLOW = 5
    WALL_BOUNDARY = 6
    SIZE = 9
