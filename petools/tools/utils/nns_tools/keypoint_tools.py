from numba import njit


@njit
def scale_predicted_kp(predictions: list, model_size: list, source_size: list):
    # predictions shape - (N, num_detected_people)
    # scale predictions
    x_scale, y_scale = [float(source_size[1]) / float(model_size[1]), float(source_size[0]) / float(model_size[0])]

    # each detection
    for i in range(len(predictions)):
        single_image_pr = predictions[i]
        # each person
        for h_indx in range(len(single_image_pr)):
            single_human_np = single_image_pr[h_indx]
            # multiply
            single_human_np[:, 0] *= x_scale
            single_human_np[:, 1] *= y_scale

    return predictions

