import random
import numpy as np
import cv2
import tensorflow as tf

def frames_from_video_file(video_path, n_frames, output_size = (172,172)):

    # Read each frame by frame
    result = []
    src = cv2.VideoCapture(str(video_path))  

    video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)

    # If the number of frames wanted is greater than the length of the video, then start from beginning
    if n_frames > video_length:
        start = 0
    else:
        # Otherwise, start at another random point within the video
        max_start = video_length - n_frames
        start = random.randint(0, max_start)

    src.set(cv2.CAP_PROP_POS_FRAMES, start)

    for _ in range(n_frames):
        ret, frame = src.read()
        if ret:
            frame = tf.image.convert_image_dtype(frame, tf.float32)
            frame = tf.image.resize_with_pad(frame, *output_size)
            result.append(frame)
        else:
            result.append(np.zeros_like(result[0]))
    src.release()
    # Ensure that the color scheme is not inverted
    result = np.array(result)[..., [2, 1, 0]]

    return result

class FrameGenerator:
    
    def __init__(self, path, n_frames):
        self.path = path
        self.n_frames = n_frames
        self.class_names = sorted(set(p.name for p in self.path.iterdir() if p.is_dir()))
        self.class_ids_for_name = dict((name, idx) for idx, name in enumerate(self.class_names))

    def get_files_and_class_names(self):
        video_paths = list(self.path.glob('*/*.mp4'))
        classes = [p.parent.name for p in video_paths] 
        return video_paths, classes

    



    def __call__(self):
        video_paths, classes = self.get_files_and_class_names()
        pairs = list(zip(video_paths, classes))
        random.shuffle(pairs)
        for path, name in pairs:
            video_frames = frames_from_video_file(path, self.n_frames) 
            #label = tf.reshape(tf.one_hot(self.class_ids_for_name[name], 3), (1,3)) # Encode labels
            label = tf.one_hot(self.class_ids_for_name[name], len(classes))
            #label = self.class_ids_for_name[name]
            yield video_frames, label