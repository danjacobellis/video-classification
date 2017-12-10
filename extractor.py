from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model
from keras.layers import Input
import numpy as np
import os
import glob
import subprocess
from datetime import datetime, timedelta
from string import capwords
import shutil
from time import time

class Extractor():
    def __init__(self, weights=None, frames_per_video=40):
        """Either load pretrained from imagenet, or load our saved
        weights from our own training."""

        self.weights = weights  # so we can check elsewhere which model
        self.frames_per_video = frames_per_video
        
        if weights is None:
            # Get model with pretrained weights.
            base_model = InceptionV3(
                weights='imagenet',
                include_top=True
            )

            # We'll extract features at the final pool layer.
            self.model = Model(
                inputs=base_model.input,
                outputs=base_model.get_layer('avg_pool').output
            )

        else:
            # Load the model first.
            self.model = load_model(weights)

            # Then remove the top so we get features not predictions.
            # From: https://github.com/fchollet/keras/issues/2371
            self.model.layers.pop()
            self.model.layers.pop()  # two pops to get to pool layer
            self.model.outputs = [self.model.layers[-1].output]
            self.model.output_layers = [self.model.layers[-1]]
            self.model.layers[-1].outbound_nodes = []

    def extract_frames(self, frames):
        x = np.zeros([len(frames), 299, 299, 3])
        for i_img, img_path in enumerate(frames):
            img = image.load_img(img_path, target_size=(299, 299))
            x[i_img] = image.img_to_array(img)
        x = preprocess_input(x)

        # Get the prediction.
        features = self.model.predict(x)

        return features

    def getLength(self, input_video):
        args = [
            'ffprobe', '-i', input_video, 
            '-show_entries', 'format=duration', 
            '-v', 'quiet', 
            '-of', 'csv=%s' % ("p=0")
        ]
        duration = subprocess.check_output(args)
        return duration
    
    def extract(self, video_file, frame_dir):
        t0 = time()
        timing = [0] * 4
        if not os.path.exists(frame_dir):
            os.makedirs(frame_dir)
        t = time()
        duration = self.getLength(video_file)
        timing[0] = time() - t
        frame_dest_name = frame_dir + os.sep + 'frame' + '-%03d.jpg'
        try:
        # EXTRACTING FRAMES HERE
            fps = self.frames_per_video/float(duration)
            args = [
                "ffmpeg", "-i", video_file, 
                "-filter:v", 'fps=' + str(fps), 
                "-vframes", str(self.frames_per_video), frame_dest_name
            ]
            t = time()
            ret = subprocess.call(args)
            timing[1] = time() - t
            if ret != 0:
                print("Failed to extract frames from %s \n" % video_file)
        except Exception as e:
            print("Failed to extract frames from %s: \n %s \n" % (video_file, e))
            print(e)
        frames = glob.glob(frame_dir + os.sep + 'frame-*.jpg')
        sequence = []
        t = time()
        sequence = self.extract_frames(frames)
#         for frame in frames:
#             features = self.extract_frame(frame)
#             sequence.append(features)
        timing[2] = time() - t
        shutil.rmtree(frame_dir)
        timing[3] = time() - t0
        timing = {
            'ffmpeg1':timing[0],
            'ffmpeg2':timing[1],
            'CNN':timing[2],
            'total':timing[3]
        }
        return sequence, timing
        