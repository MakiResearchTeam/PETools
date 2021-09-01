from petools.gpu import PosePredictor
from petools.tools.utils.visual_tools import draw_skeletons_on_image

from tqdm import tqdm
from .connect_two_videos import VideoReader, VideoWriter


def create_video_w_model(
        path_to_pb: str, path_to_config: str,
        video_read_path: str, video_save_path: str,
        path_to_pb_cor: str = None, path_to_pb_classifier: str = None,
        path_to_classifier_config: str = None, gpu_id: str = None,
        draw_pose_name: bool = False, pose_name_position: tuple = (100, 100),
        draw_pose_conf: bool = False, pose_conf_position: tuple = (120, 120),
        fps=20, **kwargs):
    """

    Parameters
    ----------
    path_to_pb : str
        Path to pb file which contain model
    path_to_config : str
        Path to config file for PosePredictor
    path_to_pb_cor : str
        Path to pose corrector, can be None, i.e. not applied
    path_to_pb_classifier : str
        Path to protobuf file with classification model
    path_to_classifier_config : str
        Path to config for classification model
    gpu_id : str
        Id of gpu, for example: '1', '0', and etc...
    video_read_path : str
        Path to read video at which human will be estimated by model
    video_save_path : str
        Path to save final video
    draw_pose_name : bool
        If true, then on video also will be pose name per frame
    pose_name_position : tuple
        Position of the pose name (X, Y)
    draw_pose_conf : bool
        If true, then confidence of pose by classificator will be shown per frame
    pose_conf_position : tuple
        Position of the conf (X, Y)
    kwargs : dict
        Additional parameters for PosePredictor

    """
    pose_predictor = PosePredictor(
        path_to_pb=path_to_pb,
        path_to_config=path_to_config,
        path_to_pb_cor=path_to_pb_cor,
        path_to_pb_classifier=path_to_pb_classifier,
        path_to_classifier_config=path_to_classifier_config,
        gpu_id=str(gpu_id),
        **kwargs
    )

    v_r = VideoReader(video_read_path)
    w_r = VideoWriter(video_save_path, fps=fps)

    iterator = tqdm(range(v_r.get_length()))

    for _ in iterator:
        s_img, is_there_more = v_r.read_frames(n=1)
        # If no more frames left - exit from loop
        if not is_there_more:
            break
        # read_frames return batched data, in our case batch_size = 1
        s_img = s_img[0]
        predictions = pose_predictor.predict(s_img)
        s_img_skeletons = draw_skeletons_on_image(
            s_img, predictions,
            draw_pose_name=draw_pose_name, pose_name_position=pose_name_position,
            draw_pose_conf=draw_pose_conf, pose_conf_position=pose_conf_position
        )
        w_r.write([s_img_skeletons])
    iterator.close()
    w_r.release()

    print(f'Creation is done! Video saved to:  {video_save_path}')

