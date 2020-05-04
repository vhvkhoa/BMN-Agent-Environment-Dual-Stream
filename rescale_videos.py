import cv2
import numpy as np
import os
import tqdm


if __name__ == '__main__':
    target_n_frames = 16 * 100
    video_dir = 'ActivityNetVideoData/'
    output_root = 'RescaledVideoData/'

    for root, _, filenames in os.walk(video_dir):
        dirname = os.path.join(output_root, os.path.relpath(root, video_dir))
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        print('Processing %s.' % dirname)
        for filename in tqdm(filenames):
            video_path = os.path.join(root, filename)
            target_video_path = os.path.join(output_root, os.path.relpath(root, video_dir), filename)
            video_path = 'v1-2/train/v_FCFSLuCZKj4.mp4'
            target_video_path = 'rescaled/v1-2/train/v_FCFSLuCZKj4.mp4'

            in_video = cv2.VideoCapture(video_path)
            num_frames = in_video.get(cv2.CAP_PROP_FRAME_COUNT)
            width = int(in_video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(in_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            _, frame = in_video.read()

            out_video = cv2.VideoWriter(
                filename=target_video_path,
                fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
                fps=30.,
                frameSize=(width, height),
                isColor=True
            )

            target_timestamps = np.linspace(0, num_frames - 1, target_n_frames)
            current_timestamp = 0
            for timestamp in target_timestamps:
                timestamp = round(timestamp)
                while timestamp > current_timestamp:
                    _, frame = in_video.read()
                    current_timestamp += 1
                out_video.write(frame)
