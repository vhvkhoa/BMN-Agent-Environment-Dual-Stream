import cv2
import numpy as np
import os
import time
from argparse import ArgumentParser


if __name__ == '__main__':
    target_n_frames = 16 * 100
    parser = ArgumentParser()
    parser.add_argument(
        '--video-root',
        type=str,
        default='ActivityNetVideoData/'
    )
    parser.add_argument(
        '--output-root',
        type=str,
        default='RescaledVideoData/'
    )
    args = parser.parse_args()

    video_root = args.video_root
    output_root = args.output_root

    for root, _, filenames in os.walk(video_root):
        dirname = os.path.join(output_root, os.path.relpath(root, video_root))
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        start_time = time.time()
        n_processed = 0

        for i, filename in enumerate(filenames):
            video_path = os.path.join(root, filename)
            target_video_path = os.path.join(
                output_root,
                os.path.relpath(root, video_root),
                os.path.splitext(filename)[0] + '.mp4'
            )

            if os.path.isfile(target_video_path):
                out_frames = cv2.VideoCapture(target_video_path).get(cv2.CAP_PROP_FRAME_COUNT)
                if out_frames == target_n_frames:
                    '''
                    print('Detected %s. %d/%d.' % (
                        os.path.basename(video_path),
                        i + 1,
                        len(filenames),
                    ))
                    '''
                    continue
            n_processed += 1

            in_video = cv2.VideoCapture(video_path)
            num_frames = int(in_video.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(in_video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(in_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

            out_video = cv2.VideoWriter(
                filename=target_video_path,
                fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
                fps=30.,
                frameSize=(width, height),
                isColor=True
            )

            in_frames = []
            for _ in range(num_frames):
                success, frame = in_video.read()
                if success:
                    in_frames.append(frame)

            target_timestamps = np.linspace(0, len(in_frames) - 1, target_n_frames)
            for timestamp in target_timestamps:
                timestamp = int(round(timestamp))
                frame = in_frames[timestamp]
                out_video.write(frame)

            in_video.release()
            out_video.release()

            processed_time = time.time() - start_time
            pred_total_time = processed_time / n_processed * (len(filenames) - i - 1) - processed_time
            h, m, s = pred_total_time // 3600, (pred_total_time % 3600) // 60, (pred_total_time % 60)
            print('Processed: %s. %d/%d. In/Out frames: %d/%d. eta: %d hours, %d minutes, %d seconds.' % (
                os.path.basename(video_path),
                i + 1,
                len(filenames),
                len(in_frames),
                int(cv2.VideoCapture(target_video_path).get(cv2.CAP_PROP_FRAME_COUNT)),
                h, m, s)
            )
