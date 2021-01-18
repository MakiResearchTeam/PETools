# Copyright (C) 2020  Igor Kilbas, Danil Gribanov
#
# This file is part of MakiPoseNet.
#
# MakiPoseNet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MakiPoseNet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <https://www.gnu.org/licenses/>.

from .constants import NUMBER_OF_KEYPOINTS


class Human:
    """
    Store keypoints of the single human

    """
    __slots__ = ('body_parts', 'score')

    def __init__(self):
        """
        Init class to store keypoints of the single human

        """
        self.body_parts = {}
        self.score = 0.0

    @staticmethod
    def _get_uidx(part_idx, idx):
        return '%d-%d' % (part_idx, idx)

    def part_count(self):
        return len(self.body_parts.keys())

    def get_max_score(self):
        return max([x.score for _, x in self.body_parts.items()])

    def to_list(self, th_hold=0.2) -> list:
        """
        Transform keypoints stored in this class to list

        Parameters
        ----------
        th_hold : float
            Threshold to store keypoints, by default equal to 0.2

        Returns
        -------
        list
            List with lenght NK * 3, where NK - Number of Keypoints,
            Where each:
            0-th element is responsible for x axis coordinate
            1-th for y axis
            2-th for visibility of the points
            If keypoint is not visible or below `th_hold`, this keypoint will be filled with zeros
        """
        list_data = []
        for i in range(NUMBER_OF_KEYPOINTS):
            take_single = self.body_parts.get(i)
            if take_single is None or take_single.score < th_hold:
                list_data += [0.0, 0.0, 0.0]
            else:
                list_data += [
                    self.body_parts[i].x,
                    self.body_parts[i].y,
                    self.body_parts[i].score,
                ]

        return list_data

    def to_dict(self, th_hold=0.2) -> dict:
        """
        Transform keypoints stored in this class to dict

        Parameters
        ----------
        th_hold : float
            Threshold to store keypoints, by default equal to 0.2

        Returns
        -------
        dict
            Dict of the keypoints,
            { NumKeypoints:   [x_coord, y_coord, score],
              NumKeypoints_1: [x_coord, y_coord, score],
              ..........................................
            }
            Where NumKeypoints, NumKeypoints_1 ... are string values responsible for index of the keypoint,
            x_coord - coordinate of the keypoint on X axis
            y_coord - coordinate of the keypoint on Y axis
            score - confidence of the neural network
            If keypoint is not visible or below `th_hold`, this keypoint will be filled with zeros
        """
        dict_data = {}
        for i in range(NUMBER_OF_KEYPOINTS):
            take_single = self.body_parts.get(i)
            if take_single is not None and take_single.score >= th_hold:
                dict_data.update({
                    str(i): [take_single.x, take_single.y, take_single.score]
                })
            else:
                dict_data.update({
                    str(i): [0.0, 0.0, 0.0]
                })

        return dict_data

    def __str__(self):
        return ' '.join([str(x) for x in self.body_parts.values()])

    def __repr__(self):
        return self.__str__()


class BodyPart:
    """
    Store single keypoints with certain coordinates and score

    """
    __slots__ = ('uidx', 'part_idx', 'x', 'y', 'score')

    def __init__(self, uidx, part_idx, x, y, score):
        """
        Init

        Parameters
        ----------
        uidx : str
            String stored number of the human and number of this keypoint
        part_idx :
        x : float
            Coordinate of the keypoint at the x-axis
        y : float
            Coordinate of the keypoint at the y-axis
        score : float
            Confidence score from neural network
        """
        self.uidx = uidx
        self.part_idx = part_idx
        self.x, self.y = x, y
        self.score = score

    def __str__(self):
        return 'BodyPart:%d-(%.2f, %.2f) score=%.2f' % (self.part_idx, self.x, self.y, self.score)

    def __repr__(self):
        return self.__str__()
