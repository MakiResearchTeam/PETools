import cv2
import numpy as np

from petools.tools.estimate_tools.human import Human


def visualize_paf(
        img,
        pafs,
        colors=((255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0))):
    img = img.copy()
    color_iter = 0
    for i in range(pafs.shape[2]):
        paf_x = pafs[:, :, i, 0]
        paf_y = pafs[:, :, i, 1]
        len_paf = np.sqrt(paf_x ** 2 + paf_y ** 2)
        for x in range(0, img.shape[0], 8):
            for y in range(0, img.shape[1], 8):
                if len_paf[x,y] > 0.1:
                    img = cv2.arrowedLine(
                        img,
                        (y, x),
                        (int(y + 9 * paf_x[x, y]), int(x + 9 * paf_y[x, y])),
                        colors[color_iter],
                        1
                    )

                    color_iter += 1

                    if color_iter >= len(colors):
                        color_iter = 0
    return img


def draw_skeleton(
        image, humans: list, connect_indexes: list, color=(255, 0, 0), thickness=2,
        draw_pose_name: bool = False, pose_name_position: tuple = (100, 100), pose_name_text_size: int = 2, pose_name_text_thick=6,
        draw_pose_conf: bool = False, pose_conf_position: tuple = (120, 120), pose_conf_text_size: int = 2, pose_conf_text_thick=6,
        pose_name_list: list = None,  pose_conf_class_list: list = None,
        draw_pose_angle: bool = False, pose_angle_position: tuple = (120, 200), pose_angle_text_size = 2, pose_angle_text_thick=6,
        draw_pose_angle_name: bool = False, pose_angle_name_position: tuple = (120, 240), pose_angle_name_text_size = 2, pose_angle_name_text_thick=6,
        pose_angle_step: tuple = (10, 0), pose_angle_name_list: list = None, pose_angle_list: list = None):
    if draw_pose_conf:
        assert len(pose_name_list) == len(humans)

    if draw_pose_conf:
        assert len(pose_conf_class_list) == len(humans)

    for indx in range(len(humans)):
        human = humans[indx]

        if isinstance(human, Human):
            data = np.array(human.to_list()).reshape(-1, 3)
        else:
            data = np.array(human).reshape(-1, 3)

        for j in range(len(connect_indexes)):
            single = connect_indexes[j]
            single_p1 = data[single[0]]
            single_p2 = data[single[1]]

            if single_p1[-1] > 1e-3 and single_p2[-1] > 1e-3:

                p_1 = (int(single_p1[0]), int(single_p1[1]))
                p_2 = (int(single_p2[0]), int(single_p2[1]))

                cv2.line(image, p_1, p_2, color=color, thickness=thickness)

        if draw_pose_name and pose_name_list is not None:
            cv2.putText(
                image, str(pose_name_list[indx]), pose_name_position, cv2.FONT_HERSHEY_SIMPLEX, pose_name_text_size, (0, 0, 255), pose_name_text_thick
            )

        if draw_pose_conf and pose_conf_class_list is not None:
            cv2.putText(
                image, str(pose_conf_class_list[indx]), pose_conf_position, cv2.FONT_HERSHEY_SIMPLEX, pose_conf_text_size, (0, 0, 255), pose_conf_text_thick
            )

        if draw_pose_angle and pose_angle_list is not None:
            start_x, start_y = pose_angle_position
            for single_angle in pose_angle_list[indx]:
                cv2.putText(
                    image, str(round(single_angle, 3)), (start_x, start_y), cv2.FONT_HERSHEY_SIMPLEX, pose_angle_text_size, (0, 0, 255), pose_angle_text_thick
                )
                start_x += pose_angle_step[0]
                start_y += pose_angle_step[1]

        if draw_pose_angle_name and pose_angle_name_list is not None:
            start_x, start_y = pose_angle_name_position
            for single_angle_name in pose_angle_name_list[indx]:
                cv2.putText(
                    image, str(single_angle_name), (start_x, start_y), cv2.FONT_HERSHEY_SIMPLEX, pose_angle_name_text_size, (0, 0, 255), pose_angle_name_text_thick
                )
                start_x += pose_angle_step[0]
                start_y += pose_angle_step[1]

    return image

