from petools.gpu import PosePredictor
from petools.tools.utils.visual_tools import draw_skeletons_on_image

from tqdm import tqdm
from connect_two_videos import VideoReader, VideoWriter


def create_video_w_model(
        path_to_pb: str, path_to_config: str,
        path_to_pb_cor: str, gpu_id: str,
        video_read_path: str, video_save_path: str):
    """

    Parameters
    ----------
    path_to_pb : str
        Path to pb file which contain model
    path_to_config : str
        Path to config file for PosePredictor
    path_to_pb_cor : str
        Path to pose corrector, can be None, i.e. not applied
    gpu_id : str
        Id of gpu, for example: '1', '0', and etc...
    video_read_path : str
        Path to read video at which human will be estimated by model
    video_save_path : str
        Path to save final video

    """
    pose_predictor = PosePredictor(
        path_to_pb=path_to_pb,
        path_to_config=path_to_config,
        path_to_pb_cor=path_to_pb_cor,
        gpu_id=str(gpu_id)
    )

    v_r = VideoReader(video_read_path)
    w_r = VideoWriter(video_save_path)

    iterator = tqdm(range(v_r.get_length()))

    for _ in iterator:
        s_img = v_r.read_frames(n=1)[0][0]
        predictions = pose_predictor.predict(s_img)
        s_img_skeletons = draw_skeletons_on_image(s_img, predictions)
        w_r.write([s_img_skeletons])
    iterator.close()
    w_r.release()

    print(f'Creation is done! Video saved to:  {video_save_path}')

