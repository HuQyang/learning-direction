import numpy as np
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc

def make_video_with_stroke_robot(images_gt, images_gen, strokes, filename):
    images_gt = unnormalize_video_data(images_gt)
    images_gen = unnormalize_video_data(images_gen)

    images_gt = draw_stroke_robot(images_gt, strokes)
    images_gen = draw_stroke_robot(images_gen, strokes)

    video = video_batch_side_by_side(images_gt, images_gen)
    save_video(video, filename,fps=2)


def draw_stroke_robot(videos, strokes):
    # video:    B x T x H x W x 3
    # stroke:   B x T x 2

    for video, stroke in zip(videos, strokes):
        for i, frame in enumerate(video):
            points = np.array(stroke[0:i+1], np.int32)
            points[:,[0,1]] = points[:,[1,0]]
            cv2.polylines(frame, [points], False, (255, 0, 0))

    return videos

def draw_stroke(videos, strokes):
    # video:    B x T x H x W x 3
    # stroke:   B x T x 2

    for video, stroke in zip(videos, strokes):
        for i, frame in enumerate(video):
            points = np.array(stroke[0:i+1], np.int32)
            # points[:,[0,1]] = points[:,[1,0]]
            cv2.polylines(frame, [points], False, (255, 0, 0))

    return videos


def video_batch_side_by_side(images_gt, images_gen, max_videos=0):
    # images_gt:    B x T x H x W x 3
    # images_gen:   B x T x H x W x 3
    assert images_gt.shape == images_gen.shape
    batch_size = images_gt.shape[0]
    max_videos = max_videos or batch_size

    images_gt = video_from_batch(images_gt[:max_videos])
    images_gen = video_from_batch(images_gen[:max_videos])

    # concatenate videos horizontally (along width)
    video = np.concatenate((images_gt, images_gen), axis=2)
    return video


def unnormalize_video_data(video):
    video = rgb2bgr(video)
    video = (video + 1.0) / 2.0
    video *= 255
    video = video.astype(np.uint8)
    return video


def save_video(video, filename, fps=5, color=True, format='XVID'):
    size = video.shape[2], video.shape[1]
    fourcc = VideoWriter_fourcc(*format)
    vid = VideoWriter(filename, fourcc, float(fps), size, color)

    for frame in video:
        vid.write(np.uint8(frame))


def video_from_batch(videos):
    # Batch of videos of size B x T x H x W x 3
    b, t, h, w, c = videos.shape
    video = np.reshape(videos, [b * t, h, w, c])
    return video


def rgb2bgr(array):
    return array[..., ::-1]


def bgr2rgb(array):
    return rgb2bgr(array)

