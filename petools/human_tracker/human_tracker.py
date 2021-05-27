import numpy as np


class HumanTracker:

    def __init__(self, image_size, square_percent=0.15):
        """

        Parameters
        ----------
        image_size : list or tuple
            List/tuple of (H, W), i.e. Height and Width of the image at which human will be appear
        square_percent : float
            Threshold for movement of the avg point

        """
        self._threash_hold_x, self._threash_hold_y = (
            image_size[1] * square_percent / 2,
            image_size[0] * square_percent / 2
        )
        # Store { human_id : avg_point }
        self._id2mean_point = dict()
        self._id_counter = 0

    def __call__(self, humans: list) -> list:
        # This array show, which element from `humans` was used
        # i.e. has id, otherwise new id will be assign
        used_h = [False]*len(humans)

        for id_h, single_avg in self._id2mean_point.items():
            # Take old avg position of points
            old_avg_x, old_avg_y = single_avg[0], single_avg[1]
            # Calc square position (left top corner and right bottom corner)
            left_top_corner_x, left_top_corner_y, right_bottom_corner_x, right_bottom_corner_y = (
                old_avg_x - self._threash_hold_x,
                old_avg_y - self._threash_hold_y,
                old_avg_x + self._threash_hold_x,
                old_avg_y + self._threash_hold_y
            )
            # Check ach human, if its in square - give `id_h` in order to track
            # This human from previous frames
            for h_indx, each_human in enumerate(humans):
                # Calc new avg point
                avg_point_new = self._mean_point_from_human(each_human)
                new_avg_x, new_avg_y = avg_point_new[0], avg_point_new[1]
                # Is this point (new avg) in square?
                if left_top_corner_x < new_avg_x < right_bottom_corner_x and \
                        left_top_corner_y < new_avg_y < right_bottom_corner_y:
                    # If in square, then
                    # This is old one point, give it old id
                    each_human.id = int(id_h)
                    used_h[h_indx] = True
                    # Update avg
                    self._id2mean_point[id_h] = avg_point_new
                    # Can be only 1 person in this square, so we can end loop
                    break

        for indx_used in range(len(used_h)):
            if not used_h[indx_used]:
                # Add id new human and calc new avg
                humans[indx_used].id = self._id_counter
                avg_point = self._mean_point_from_human(humans[indx_used])
                # Assign id and avg point to it
                self._id2mean_point[str(self._id_counter)] = avg_point
                # Update counter, in order to handle unique values of id
                self._id_counter += 1

        return humans

    def _mean_point_from_human(self, human):
        """
        Calculate avg point from human

        """
        h_np = human.to_np()
        visible_h_np = h_np[h_np[:, -1] > 1e-3]
        return np.mean(visible_h_np[:, :-1], axis=0)

