import cv2
import numpy as np
from tqdm import tqdm


class VideoReader:
    def __init__(self, video_path):
        """
        A utility for batching frames from a video file.
        Parameters
        ----------
        video_path : str
            Path to the video file.
        """
        self._path = video_path
        self._video = None
        self._last_frame = None
        self.reset()

    def reset(self):
        """
        Resets the state of the reader, so that it can read frames again.
        Returns
        -------
        """
        self._video = cv2.VideoCapture(self._path)
        assert self._video.isOpened(), f'Could not open video with path={self._path}'
        self._last_frame = None

    def get_length(self):
        """
        Returns
        -------
        int
            Number of frames in the video.
        """
        return int(self._video.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_frame_size(self):
        """
        Returns
        -------
        (int, int)
            Frame height and width.
        """
        height = int(self._video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(self._video.get(cv2.CAP_PROP_FRAME_WIDTH))
        return height, width

    def get_fps(self):
        """
        Returns
        -------
        int
            Fps number.
        """
        return self._video.get(cv2.CAP_PROP_FPS)

    def is_opened(self):
        """
        Returns
        -------
        bool
            True if the video file is opened. False otherwise.
        """
        return self._video.isOpened()

    def close(self):
        """
        Closes the video file.
        """
        self._video.release()

    def read_frames(self, n=1, transform=None) -> (list, bool):
        """
        Reads a batch of frames and returns them packed in list.
        If there are not enough enough frames for the batch,
        it will pad the missing frames with the last one and also return False in the flag.
        This method guaranties that the internal video reader will be released as soon as all
        frames are read.
        Parameters
        ----------
        n : int
            How many frames to read.
        transform : python function
            Will be applied to each frame.
        Returns
        -------
        list
            List of n frames.
        bool
            Flag that shows if there are frames left in the video.
        """
        assert self._video.isOpened(), 'There are no frames left. Please, reset the video reader.'

        if transform is None:
            transform = lambda x: x

        frames = []
        # Transform and add to the list the last frame if it is present
        if self._last_frame is not None:
            frame = transform(self._last_frame)
            frames.append(frame)
            self._last_frame = None

        for _ in range(n - len(frames)):
            ret, frame = self._video.read()
            if not ret:
                print('Ran out of frames.')
                self._video.release()
                break

            frames.append(transform(frame))

        assert len(frames) != 0, 'There are no frames left. Please, reset the video reader. (video is opened, for devs)'

        # Pad lacking frames
        if len(frames) != n:
            k = len(frames)
            to_add = n - k
            frames = frames + [frames[-1]] * to_add

        # Sanity check
        assert len(frames) == n, f'Number of frames={len(frames)} is not equal to the requested amount={n}'

        # This is used to check whether there are frames left.
        ret, self._last_frame = self._video.read()
        if not ret:
            self._video.release()

        return frames, self._video.isOpened()

    def get_iterator(self, batch_size, transform=None):
        """
        Creates an iterator that yields batches of frames from the video.
        Parameters
        ----------
        batch_size : int
            The batch size.
        transform : python function
            Will be applied to each frame in the batch.
        Returns
        -------
        python iterator
        """

        frame_batch, has_frames = self.read_frames(batch_size, transform=transform)
        while has_frames:
            yield frame_batch
            frame_batch, has_frames = self.read_frames(batch_size, transform=transform)
        yield frame_batch


class VideoWriter:
    def __init__(
            self, video_path, fps=20):
        """
        Parameters
        ----------
        video_path : str
        fps : int

        """
        self._video_path = video_path
        self._fps = fps
        self._video = None

    def _init(self, frame_size):
        height, width = frame_size
        self._video = cv2.VideoWriter(
            self._video_path, cv2.VideoWriter_fourcc(*'mp4v'), self._fps,
            (width, height)
        )

    def write(self, images):
        """

        Parameters
        ----------
        images : list
            List of images (ndarrays).

        """
        if self._video is None:
            h, w, c = images[0].shape
            self._init((h, w))

        for image in images:
            self._video.write(image)

    def release(self):
        self._video.release()


class ConnectTwoVideos:

	def __init__(self, path_video_1, path_video_2, text_1="", text_2="", pos1=(120, 400), pos2=(120, 400)):
		self._path1 = path_video_1
		self._path2 = path_video_2
		self._video_r_1 = VideoReader(self._path1)
		self._video_r_2 = VideoReader(self._path2)
		self._text1 = str(text_1)
		self._text2 = str(text_2)

		self._pos1 = pos1
		self._pos2 = pos2

	def connect(self, path_save_to, fps=20, axis=1):

		video_wr = VideoWriter(path_save_to, fps)

		iterator = tqdm(
			range(
				min(self._video_r_1.get_length(), self._video_r_2.get_length())
			)
		)

		for _ in iterator:
			# Take single image
			img1 = self._video_r_1.read_frames()[0][0]
			cv2.putText(img1, self._text1, self._pos1, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 6)

			img2 = self._video_r_2.read_frames()[0][0]
			cv2.putText(img2, self._text2, self._pos2, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 6)

			concat_img = np.concatenate([img1, img2], axis=axis).astype(np.uint8)
			video_wr.write([concat_img])

		iterator.close()
		print('all done!')

