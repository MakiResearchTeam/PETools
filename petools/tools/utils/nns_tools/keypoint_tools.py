

def scale_predicted_kp(predictions: list, model_size: tuple, source_size: tuple):
    # predictions shape - (N, num_detected_people)
    # scale predictions
    scale = [source_size[1] / model_size[1], source_size[0] / model_size[0]]

    # each detection
    for i in range(len(predictions)):
        single_image_pr = predictions[i]
        # each person
        for h_indx in range(len(single_image_pr)):
            single_human_np = single_image_pr[h_indx]
            # multiply
            single_human_np[:, 0] *= scale[0]
            single_human_np[:, 0] *= scale[1]

    return predictions

