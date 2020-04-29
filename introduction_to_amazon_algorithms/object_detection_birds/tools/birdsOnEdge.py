#*****************************************************
#                                                    *
# Copyright 2019 Amazon.com, Inc. or its affiliates. *
# All Rights Reserved.                               *
#                                                    *
#*****************************************************
from threading import Thread, Event, Timer
import os
import json
import numpy as np
import greengrasssdk
import sys
import datetime
import time
import awscam
import cv2
import urllib
import zipfile
import mo

# Create a greengrass core sdk client
client = greengrasssdk.client('iot-data')

# The information exchanged between IoT and clould has a topic and a
# message body. This is the topic used to send messages to cloud.
iot_topic = '$aws/things/{}/infer'.format(os.environ['AWS_IOT_THING_NAME'])

client.publish(topic=iot_topic, payload='At start of lambda function')

# boto3 is not installed on device by default. Install it if it is not
# already present.

boto_dir = '/tmp/boto_dir'
if not os.path.exists(boto_dir):
    os.mkdir(boto_dir)
    urllib.urlretrieve('https://s3.amazonaws.com/dear-demo/boto_3_dist.zip',
                        '/tmp/boto_3_dist.zip')
    client.publish(topic=iot_topic, payload='Extracting boto3 distribution...')
    with zipfile.ZipFile('/tmp/boto_3_dist.zip', 'r') as zip_ref:
        zip_ref.extractall(boto_dir)
client.publish(topic=iot_topic, payload='Adding boto to path...')
sys.path.append(boto_dir)

import boto3

client.publish(topic=iot_topic, payload='Completed import of boto3')

class LocalDisplay(Thread):
    """ Class for facilitating the local display of inference results
        (as images). The class is designed to run on its own thread. In
        particular the class dumps the inference results into a FIFO
        located in the tmp directory (which lambda has access to). The
        results can be rendered using mplayer by typing:
        mplayer -demuxer lavf -lavfdopts format=mjpeg:probesize=32 /tmp/results.mjpeg
    """
    def __init__(self, resolution):
        """ resolution - Desired resolution of the project stream """
        # Initialize the base class, so that the object can run on its own
        # thread.
        super(LocalDisplay, self).__init__()
        # List of valid resolutions
        RESOLUTION = {'1080p' : (1920, 1080), '720p' : (1280, 720), '480p' : (858, 480)}
        if resolution not in RESOLUTION:
            raise Exception("Invalid resolution")
        self.resolution = RESOLUTION[resolution]
        # Initialize the default image to be a white canvas. Clients
        # will update the image when ready.
        self.frame = cv2.imencode('.jpg', 255*np.ones([640, 480, 3]))[1]
        self.stop_request = Event()

    def run(self):
        """ Overridden method that continually dumps images to the desired
            FIFO file.
        """
        # Path to the FIFO file. The lambda only has permissions to the tmp
        # directory. Pointing to a FIFO file in another directory
        # will cause the lambda to crash.
        result_path = '/tmp/results.mjpeg'
        # Create the FIFO file if it doesn't exist.
        if not os.path.exists(result_path):
            os.mkfifo(result_path)
        # This call will block until a consumer is available
        with open(result_path, 'w') as fifo_file:
            while not self.stop_request.isSet():
                try:
                    # Write the data to the FIFO file. This call will block
                    # meaning the code will come to a halt here until a consumer
                    # is available.
                    fifo_file.write(self.frame.tobytes())
                except IOError:
                    continue

    def set_frame_data(self, frame):
        """ Method updates the image data. This currently encodes the
            numpy array to jpg but can be modified to support other encodings.
            frame - Numpy array containing the image data of the next frame
                    in the project stream.
        """
        ret, jpeg = cv2.imencode('.jpg', cv2.resize(frame, self.resolution))
        if not ret:
            raise Exception('Failed to set frame data')
        self.frame = jpeg

    def join(self):
        self.stop_request.set()

def greengrass_infinite_infer_run():
    """ Entry point of the lambda function"""

    client.publish(topic=iot_topic, payload='Start of run loop...')

    try:
        # This object detection model is implemented as single shot detector (ssd), since
        # the number of labels is small we create a dictionary that will help us convert
        # the machine labels to human readable labels.
        model_type = 'ssd'

        output_map = {}
        with open('classes.txt') as f:
            for line in f:
                (key, val) = line.split()
                output_map[int(key)] = val

        client.publish(topic=iot_topic, payload='Classes to be detected: ' + str(output_map))

        # Create a local display instance that will dump the image bytes to a FIFO
        # file that the image can be rendered locally.
        local_display = LocalDisplay('480p')
        local_display.start()

        # The height and width of the training set images
        input_height = 512
        input_width  = 512

        # Load the model onto the GPU.

        # optimize the model
        client.publish(topic=iot_topic, payload='Optimizing model...')
        ret, model_path = mo.optimize('deploy_ssd_resnet50_512', input_width, input_height)

        # load the model
        client.publish(topic=iot_topic, payload='Loading model...')
        model = awscam.Model(model_path, {'GPU': 1})

        client.publish(topic=iot_topic, payload='Custom object detection model loaded')

        # Set the threshold for detection
        detection_threshold = 0.40

        # Do inference until the lambda is killed.
        while True:
            # Get a frame from the video stream
            ret, frame = awscam.getLastFrame()
            if not ret:
                raise Exception('Failed to get frame from the stream')
            # Resize frame to the same size as the training set.
            frame_resize = cv2.resize(frame, (input_height, input_width))

            # Run the images through the inference engine and parse the results using
            # the parser API.  Note it is possible to get the output of doInference
            # and do the parsing manually, but since it is a ssd model,
            # a simple API is provided.
            client.publish(topic=iot_topic, payload='Calling inference on next frame...')
            parsed_inference_results = model.parseResult(model_type,
                                                         model.doInference(frame_resize))

            # Compute the scale in order to draw bounding boxes on the full resolution
            # image.
            yscale = float(frame.shape[0]) / float(input_height)
            xscale = float(frame.shape[1]) / float(input_width)

            # Dictionary to be filled with labels and probabilities for MQTT
            cloud_output = {}
            # Get the detected objects and probabilities
            default_color = (255, 165, 20)
            i = 0
            for obj in parsed_inference_results[model_type]:
                if obj['prob'] > detection_threshold:
                    # Add bounding boxes to full resolution frame
                    xmin = int(xscale * obj['xmin'])
                    ymin = int(yscale * obj['ymin'])
                    xmax = int(xscale * obj['xmax'])
                    ymax = int(yscale * obj['ymax'])

                    # See https://docs.opencv.org/3.4.1/d6/d6e/group__imgproc__draw.html
                    # for more information about the cv2.rectangle method.
                    # Method signature: image, point1, point2, color, and tickness.
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), default_color, 10)
                    # Amount to offset the label/probability text above the bounding box.
                    text_offset = 15
                    # See https://docs.opencv.org/3.4.1/d6/d6e/group__imgproc__draw.html
                    # for more information about the cv2.putText method.
                    # Method signature: image, text, origin, font face, font scale, color,
                    # and thickness
                    cv2.putText(frame, "{}: {:.2f}%".format(output_map[obj['label']],
                                                               obj['prob'] * 100),
                                (xmin, ymin-text_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 165, 20), 6)
                    # Store label and probability to send to cloud
                    cloud_output[output_map[obj['label']]] = obj['prob']

            # Set the next frame in the local display stream.
            local_display.set_frame_data(frame)
            # Send results to the cloud
            client.publish(topic=iot_topic, payload=json.dumps(cloud_output))
    except Exception as ex:
        client.publish(topic=iot_topic, payload='Error in object detection lambda: {}'.format(ex))

# Execute the function above
greengrass_infinite_infer_run()

# This is a dummy handler and will not be invoked
# Instead the code above will be executed in an infinite loop for our example
def function_handler(event, context):
    return
