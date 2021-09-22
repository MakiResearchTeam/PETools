from queue import Queue
from copy import deepcopy

PARENT_CHILDREN_MAPPING = {}


def find_child_points(point_ind: int, connect_indices: list) -> list:
    """
    Finds neighboring points (their indices) of the points with `point_ind` index.

    Parameters
    ----------
    point_ind : int
        Index of the point which neighbours to find.
    connect_indices : list
        Contains pairs of points indices (from, to) depicting a skeleton graph structure.

    Returns
    -------
    list
        Indices of neighbouring points.
    """
    cached, children = PARENT_CHILDREN_MAPPING.get(point_ind, (False, []))
    if cached:
        return children

    for p1_ind, p2_ind in connect_indices:
        if point_ind == p1_ind:
            children.append(p2_ind)

    PARENT_CHILDREN_MAPPING[point_ind] = True, children
    return children


def coherence_check(points, root_points: tuple, connect_indices: list, conf_threshold=1e-3):
    """
    Masks out all the points (sets confidence to 0.0) that are located after the roots points in
    the skeleton graph.

    Parameters
    ----------
    points : array like
        A human points.
    root_points : tuple
        List of indices of the root points. Those points are used as the start for breadth search for masking
        *unconnected points*.
    connect_indices : list
        Contains pairs of points indices (from, to) depicting a skeleton graph structure.
    conf_threshold : float
        A minimum confidence a point must have to be visible.

    Returns
    -------
    points : same as points in args
        Modified points.
    """
    points = deepcopy(points)

    for i in range(len(root_points)):
        if points[root_points[i]][2] > conf_threshold:
            root_point = root_points[i]

            frontier = Queue()
            is_absent = {root_point: False}
            frontier.put(root_point)
            while not frontier.empty():
                current = frontier.get()
                parent_is_absent = is_absent[current]
                for child in find_child_points(current, connect_indices):
                    frontier.put(child)
                    assert is_absent.get(child) is None, \
                        'Bad connect_indices, the skeleton graph must be acyclic. ' \
                        f'A point with ind={child} can be visited twice.'
                    child_conf = points[child][2]
                    is_absent[child] = parent_is_absent or child_conf < conf_threshold
                    if is_absent[child]:
                        points[child][2] = 0.0

    return points

