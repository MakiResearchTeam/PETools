from numba import njit

@njit
def scale_predicted_kp(predictions: list, model_size: tuple, source_size: tuple):
    # predictions shape - (N, num_detected_people)
    # scale predictions
    scale = [source_size[1] / model_size[1], source_size[0] / model_size[0]]
    for h_indx in range(len(predictions)):
        single_human_np = predictions[h_indx]
        # multiply
        single_human_np[:, 0] *= scale[0]
        single_human_np[:, 1] *= scale[1]

    return single_image_pr

