import argparse
import os
import time
import numpy as np
import operator
import cv2
from image_processor import ImageProcessor
from PIL import Image, ImageDraw


class ObjectDetection:
    """
    This class is used for detect, crop objects in a image using the DNN mobilenetsv2 API for object detection.

    """
    # Set up camera constants
    # Input resolution

    IM_WIDTH = 320
    IM_HEIGHT = 240

    # Output resolutions
    shape = (2560, 1920)    # Taking the 5 mpx resolution 2560 x 1920 as reference

    # Scale relation between low and high resolution
    scaleX = 8
    scaleY = 8

    # Shape to rescale the high resolution.
    new_shape = (int(shape[0] / scaleX), int(shape[1] / scaleY))

    def __init__(self):
        """
        Constructor.

        This instantiates the Object detector from the Object Detection API from Google
        
        :
        """
        self.detect = ImageProcessor()
        self.detect.setup()
        self.index_to_string = {
            3: 'car',
            6: 'bus',
            8: 'truck',
            1: 'person'
        }

    @staticmethod
    def get_centroid(x, y, w, h):
        """
        This calculates a centroid from the four points of a rectangle as input.
        :param x: low x position
        :param y: low y position
        :param w: high x position
        :param h: high y position
        :return: tuple as centroid position of the input box
        """
        x1 = int(w / 2)
        y1 = int(h / 2)
        cx = x + x1
        cy = y + y1
        return cx, cy

    def _detect_objects(self, image_array, vector=False, desired_score=0.3):
        """
        Creates input for the object detection
        param path_to_image: Absolute path to image of interest
        :param image_array: numpy array of the target image
        :param vector: Bool if True return detection vector
        :output Dictionary with the results of the detection in the image
        """

        # Obtain detection for image array
        (boxes, scores, classes, num) = self.detect.detect(image_array)
        if vector is True:

            # Create a placeholder for return detections
            response = {'results': [
                    {'prediction': [],
                     'vector': (boxes, scores, classes, num)}
                ]
            }

        else:
            # Create a placeholder for return detections
            response = {'results': [
                    {'prediction': []}
                ]
            }

        # Filter just car detections.
        for i, b in enumerate(boxes[0]):
            #        person  1       car    3                bus   6               truck   8
            if classes[0][i] == 3 or classes[0][i] == 6 or classes[0][i] == 8:
                if scores[0][i] >= desired_score:
                    x0 = int(boxes[0][i][3] * image_array.shape[1])
                    y0 = int(boxes[0][i][2] * image_array.shape[0])

                    x1 = int(boxes[0][i][1] * image_array.shape[1])
                    y1 = int(boxes[0][i][0] * image_array.shape[0])

                    response['results'][0]['prediction'].append({
                        'coord': {
                            'xmin': x0, 'ymin': y0,
                            'xmax': x1, 'ymax': y1
                        },
                        'class': self.index_to_string[classes[0][i]],
                        'prob': scores[0][i]
                    })
        return response

    @staticmethod
    def _filter_objects(response):
        """
        This grab the results from object detection and return json and
        optionally draw the results in the input image
        param response: Dictionary like with the detection resutls
        param draw_image: Boolean, if True, draw the detection result in target image
        """

        predictions = response["results"][0]["prediction"]

        if len(predictions) > 0:
            # Placeholder for the results 
            image_paths_and_detection = {}
            image_paths_and_detection['detections'] = []

            # Aux list for keep track of areas
            possible_detections = []
            for _, pred in enumerate(predictions):

                x, y = pred['coord']["xmin"], pred['coord']["ymin"]
                w, h = pred['coord']["xmax"], pred['coord']["ymax"]
                area = (w-x)*(h-y)

                # centroid = self.get_centroid(x, y, w, h)

                # Draw simple rectangle over detected object.
                detection = {
                    'class': pred['class'],
                    'prob': pred['prob'],
                    'area': area,
                    'cord': (x, y, w, h),
                    }
                # append this to list comparator
                possible_detections.append(detection)
            possible_areas = []
            # Append all areas to possible areas list
            for possible in possible_detections:
                possible_areas.append(possible['area'])

            # Get the max area for this list
            _, max_area = max(enumerate(possible_areas), key=operator.itemgetter(1))

            detected_car = {}
            for possible in possible_detections:

                if possible['area'] == max_area:
                    # Rename this possible as official detection
                    official_detection = possible

                    # Draw this official detection to image
                    x, y, w, h = official_detection['cord']
                    detected_car['coord'] = (x, y, w, h)
                    detected_car['class'] = possible['class']
                    detected_car['prob'] = possible['prob']

            return detected_car
        else:
            return {}

    def roi_results(self, path_to_image):
        """
        This Cuts the input image using the detection results, for accomplish this resizing
        the 5 MP image size is used, later this is resized to a low resolution image and feeded
        to a object detection algorithm, the results of this are used to cut the 5MP resized
        image, and this cuts are saved in the same directory where the path_to_image lives
        with the end name, *_crop.jpg, we also return a dict with the object detected location on
        the 5 MP image.
        :param path_to_image: Absolute path to target image to detect
        :return: I/O writo to disk the croped image and return a dict with the location of the
                object in the 5 mp image.
        """

        image_array = cv2.imread(path_to_image)
        image_array_resized_high = cv2.resize(image_array, ObjectDetection.shape)
        image_array_resized_low = cv2.resize(image_array, ObjectDetection.new_shape)

        # Detect objects
        results = self._detect_objects(image_array_resized_low)

        detected_car = self._filter_objects(results)
        output_json = {}
        if len(detected_car) > 0:
            x, y, w, h = detected_car['coord']
            class_type = detected_car['class']
            prob = detected_car['prob']

            x0 = x*ObjectDetection.scaleX
            y0 = y*ObjectDetection.scaleY

            x1 = w*ObjectDetection.scaleX
            y1 = h*ObjectDetection.scaleY

            output_json['class'] = class_type
            output_json['prob'] = prob
            output_json['coord'] = (x0, y0, x1, y1)

            # Cut image in HR image
            cropped_image_name = '{}_crop.jpg'.format(path_to_image[:path_to_image.rfind('.')])
            cropped_original_raw = image_array_resized_high[y1: y0, x1: x0]
            cv2.imwrite(cropped_image_name, cropped_original_raw)

            return output_json

    def detection(self, path_to_image, draw=False):
        """
        This method detect desired objects on image and return a list of this detections
        in the form of dicts, it takes the low resolution image and makes the detections on it,
        later this is feeded into the detection algorithm and return the detections, this also draws this
        detections into the low resolution frame that was feeded.

        :param path_to_image: Absolute path to target image to detect
        :param draw: Boolean, if True draw rectangle on image and save in image
                    target dir.
        :return: List of dicts containing the detections.
        """
        image_array = cv2.imread(path_to_image)
        image_array_resized_low = cv2.resize(image_array, ObjectDetection.new_shape)

        response = self._detect_objects(image_array_resized_low, vector=True)

        boxes, scores, classes, num = response['results'][0]['vectors']
        if draw is True:
            # Anotate the image with the object detection modules
            frame = detect.annotate_image(image_array_resized_low, boxes, classes, scores)

            # Desired name and path to save.
            path_to_new_image = '{}_crop.jpg'.format(path_to_image[:path_to_image.rfind('.')])

            # Save to this path.
            cv2.imwrite("{}_detected.jpg".format(path_to_new_image), frame)

        return response["results"][0]["prediction"]


if __name__ == '__main__':

    # For pass argument file
    parser = argparse.ArgumentParser(description='Add folder to process')
    parser.add_argument('-f', '--checkImage', default = None, type=str, help="Add path to the folder to check")

    args = parser.parse_args()

    if args.checkImage != None:
        rutaDeTrabajo = args.checkImage
        print('Ruta a limpiar: {}'.format(rutaDeTrabajo))
    else:
        print('No se introdujo folder a revisar')


    # Instantiate detection class
    detect = ObjectDetection()

    # Load the images into the target directory
    fotografias = [f for f in os.listdir(rutaDeTrabajo) if '.jpg' in f]

    print('Analizando {} imagenes'.format(len(fotografias)))

    for fotografia in fotografias:
        path_to_original_image = rutaDeTrabajo+'/'+fotografia

        tiempoMedicion = time.time()

        results = detect.roi_results(path_to_original_image)
        #results = detect.object_get_results(path_to_original_image, detect=True)
        #results = detect._detect_objects(path_to_original_image)

        print('TOTAL TIME IS...: ', time.time() - tiempoMedicion)
        print(results)