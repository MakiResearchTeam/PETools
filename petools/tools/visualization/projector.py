import numpy as np
from scipy.spatial.transform import Rotation as R


def dict_to_numpy(points3d):
    return np.array(list(points3d.values()))


def convert_to_3dnp(preds, centering_point=None):
    humans = preds['humans']
    ids = []
    points3d_list = []
    centers = []
    for id, points2d, points3d, pose_info, angles_info in humans:
        ids.append(id)
        points3d_list.append(dict_to_numpy(points3d))
        if centering_point:
            centers.append(points2d[f'p{centering_point}'])
        else:
            centers.append(points2d['p22'])
    return ids, points3d_list, centers


def numpy_to_dict2d(points):
    dict_points = {}
    for i, (x, y, p) in enumerate(points):
        if i == 22:
            dict_points[f'p{i}'] = [x, y, 1.0]
            continue

        dict_points[f'p{i}'] = [x, y, p]

    return dict_points


def create_fake_preds(ids, dicts_points2d):
    humans = []
    for id, dict_points2d in zip(ids, dicts_points2d):
        humans.append((id, dict_points2d, None, ('None', 0.0)))

    return {'humans': humans}


class Projector:
    def __init__(
            self, n_points=23, scale=1.0, shift=None, z_shift: float = None,
            z_far=100., z_near=0.1, fov=45.0, h=1000, w=1000,
            rotation_info: dict = None, centering_point: int = None, perspective_divide=False
    ):
        """
        Projects 3D points so that they can be drawn using standard visualization tools.

        Parameters
        ----------
        n_points : int
            Number of points in the human skeleton.
        scale : float
            The projected 3d points will be multiplied with this number.
        shift : array-like
            It will be added to the x and y coordinates of the projected points.
        z_shift : float, optional
            Shift along Z axis.
        z_far : float
            The farthest visible point.
        z_near : float
            The nearest visible point.
        fov : float
            Field of view.
        h : int
            Height of the image to draw on. Used to compute aspect ratio.
        w : int
            Width of the image to draw on. Used to compute aspect ratio.
        rotation_info : dict, optional
            A dictionary which contains parameters from `scipy.spatial.transform.from_euler` method.
            Example: {'seq': 'xyz', angles: [90, 0, 0], 'degrees': True }
        centering_point : int, optional
            If provided, all the 3d poses will be centered around the `centering_point`th point of the
            corresponding 2d pose.
        perspective_divide : bool
            Whether to perform perspective divide when projecting the points.
        """
        self.projmat = self.init_projmat(z_far=z_far, z_near=z_near, fov=fov, h=h, w=w)
        self.h = h
        self.w = w
        self.scale = scale
        self.shift = np.asarray(shift)
        self.rotation_matrix = None
        if rotation_info:
            self.rotation_matrix = R.from_euler(**rotation_info).as_matrix().astype('float32')

        if shift is None:
            self.shift = np.array([scale / 2, scale / 2], dtype='float32')

        self.centering_point = centering_point
        self.z_shift = z_shift
        self.perspective_divide = perspective_divide
        self.buffer = np.ones((n_points, 4), dtype='float32')

    def init_projmat(self, z_far=100., z_near=0.1, fov=45.0, h=1000, w=1000):
        f = 1 / np.tan(fov / 2 / 180 * np.pi)
        a = h / w
        q = z_far / (z_far - z_near)
        return np.array([
            [f * a, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, q, 1],
            [0, 0, -z_near * q, 0]
        ], dtype='float32')

    def project_points(self, points):
        self.buffer[:, :3] = points
        proj_points = self.buffer.dot(self.projmat)
        if self.perspective_divide:
            proj_points /= proj_points[:, -1, None] + 1e-6
        proj_points[:, 0] *= self.w / 2
        proj_points[:, 1] *= self.h / 2
        return proj_points[:, :3]

    def project(self, preds: dict, return_ids=True, return_fake_preds=False):
        """
        Do projection of the 3d points in `preds`.

        Parameters
        ----------
        preds : dict
            A dictionary returned by the PosePredictor.
        return_ids : bool, default=True

        return_fake_preds

        Returns
        -------
        If return_fake_preds=True, returns a dictionary with the same structure as the one returned by PosePredictor,
        but the 2d points are actually the projected 3d points. Human IDs are preserved.
        If return_ids=True it returns a list of projected 3d humans and a list of the corresponding IDs.
        Otherwise only the list of projected 3d humans is returned.
        """
        ids, humans3dnp, centers = convert_to_3dnp(preds, self.centering_point)
        projected_humans = []
        for human3d, center in zip(humans3dnp, centers):
            human3d[:, :3] /= 1000 * 2.5

            # Rotate the human
            if isinstance(self.rotation_matrix, np.ndarray):
                human3d[:, :3] = human3d[:, :3].dot(self.rotation_matrix)
            # Set the confidence of the hip point to 1.0, since initially it is always 0.0
            human3d[22, 3] = 1.0
            if self.z_shift:
                human3d[:, 2] += self.z_shift

            human3d[:, :3] = self.project_points(human3d[:, :3])

            # Perform scaling and shifting
            human3d[:, :3] *= self.scale
            if self.centering_point:
                human3d[:, :2] -= human3d[self.centering_point, :2]
                human3d[:, :2] += center[:2]
            else:
                human3d[:, :2] += self.shift

            # Set the confidences to stand next to the coordinates
            human3d[:, 2] = human3d[:, 3]
            projected_humans.append(human3d[:, :3])

        if return_fake_preds:
            human_dicts = []
            for projected_human in projected_humans:
                human_dicts.append(numpy_to_dict2d(projected_human))

            return create_fake_preds(ids, human_dicts)

        if return_ids:
            return projected_humans, ids

        return projected_humans
