from tqdm import tqdm

from petools.core import PosePredictorInterface
from .skeleton_video_writer import SkeletonDrawer
from .video_reader import VideoReader

DEFAULT_BATCH_SIZE = 1


def video_predict_writer(video_path, model: PosePredictorInterface, result_path='result.mp4', fps=20):
    norm_iter = VideoReader(video_path).get_iterator(DEFAULT_BATCH_SIZE)
    orig_iter = VideoReader(video_path).get_iterator(DEFAULT_BATCH_SIZE)

    video_writer = SkeletonDrawer(result_path, fps=fps)

    for norm_batch, orig_batch in tqdm(zip(norm_iter, orig_iter)):
        pred_humans_batch = [model.predict(norm_batch[0])]
        video_writer.write(orig_batch, pred_humans_batch)
    video_writer.release()

