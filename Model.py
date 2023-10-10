import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import imghdr
from tempfile import NamedTemporaryFile

from PIL import Image, ImageDraw, ImageFont
import cv2

weights_path = './files/yolov3.weights'
configuration_path = './files/yolov3.cfg'
labels_path = './files/coco.names'
img_path = "./images/city_scene.jpg"

with open(labels_path, 'r') as file:
    labels = file.read().strip().split('\n')

class YoloModel():
    
    def __init__(self, weights_url, configuration_path, labels_path, prob_min=0.5, threshold=0.3):
        self.name = "YOLO V3"
        self.weights_path = self.download_weights(weights_url)
        self.configuration_path = configuration_path
        self.labels_path = labels_path
        self.probability_minimum = prob_min
        self.threshold = threshold

        with open(labels_path, 'r') as file:
            self.labels = file.read().strip().split('\n')

        self.network = cv2.dnn.readNetFromDarknet(configuration_path, weights_path)
        self.layers_names_output = self.network.getUnconnectedOutLayersNames()
        self.layers_names_all = self.network.getLayerNames()

    def download_weights(self, weights_url):
        response = requests.get(weights_url)
        if response.status_code == 200:
            temp_weights_file = NamedTemporaryFile(delete=False)
            temp_weights_file.write(response.content)
            temp_weights_file.close()
            return temp_weights_file.name
        else:
            raise Exception("Falha ao baixar o arquivo de pesos.")
    
    def show_architecture(self):
        layer_names = self.network.getLayerNames()
        print(f"=== {self.name} Architecture ===")

        for layer_name in layer_names:
            print(layer_name)

        print(f"=== {self.name} Architecture ===")
        print(f"Total de Camadas: {len(layer_names)}")

    def show_image_path(self, path):
            image = cv2.imread(path)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.show()

    def show_image(self, img):
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.show()

    def pre_process_image(self, path, model_image_size=(416, 416)):
            image_type = imghdr.what(path)
            image = cv2.imread(path)
            resized_image = cv2.resize(image, model_image_size)
            image_data = resized_image.astype(np.float32) / 255.0
            image_data = np.expand_dims(image_data, axis=0)
            return image, image_data, image_type, resized_image

    def calculate_iou(self, box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        x_intersection = max(0, min(x2_1, x2_2) - max(x1_1, x1_2))
        y_intersection = max(0, min(y2_1, y2_2) - max(y1_1, y1_2))

        intersection_area = x_intersection * y_intersection

        area_box1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area_box2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area_box1 + area_box2 - intersection_area

        iou = intersection_area / union_area

        return iou

    def non_max_suppression(self, boxes, confidences):
        if len(boxes) == 0:
            return []

        selected_indices = []

        boxes = np.array(boxes)

        sorted_indices = np.argsort(confidences)[::-1]

        while len(sorted_indices) > 0:
            i = sorted_indices[0]
            selected_indices.append(i)

            iou = [self.calculate_iou(boxes[i], boxes[j]) for j in sorted_indices[1:]]

            filtered_indices = [j for j in range(1, len(sorted_indices)) if iou[j - 1] <= self.threshold]

            sorted_indices = np.array(sorted_indices)[filtered_indices]

        return selected_indices

    def detect(self, output, h, w):

        bounding_boxes = []
        confidences = []
        class_numbers = []
        bounding_boxes2 = []

        for result in output:
            for detection in result:

                scores = detection[5:]

                class_current = np.argmax(scores)

                confidence_current = scores[class_current]

                if confidence_current > self.probability_minimum:
                    box_current = detection[0:4] * np.array([w, h, w, h])

                    x_center, y_center, box_width, box_height = box_current.astype('int')
                    x_min = int(x_center - (box_width / 2))
                    y_min = int(y_center - (box_height / 2))
                    x_max = int(x_center + (box_width / 2))
                    y_max = int(x_center + (box_height / 2))

                    bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                    bounding_boxes2.append([x_min, y_min, x_max, y_max])
                    confidences.append(float(confidence_current))
                    class_numbers.append(class_current)

        return bounding_boxes, confidences, class_numbers, bounding_boxes2

    def getNewImage(self, imagem, bounding_boxes, class_numbers, confidences, results, title=None):
        results = np.array(results)
        np.random.seed(42)
        colours = np.random.randint(0, 255, size=(len(self.labels), 3), dtype='uint8')

        if len(results) > 0:
            for i in results.flatten():
                x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
                box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

                colour_box_current = [int(j) for j in colours[class_numbers[i]]]

                cv2.rectangle(imagem, (x_min, y_min), (x_min + box_width, y_min + box_height),
                                colour_box_current, 5)

                text_box_current = '{}: {:.4f}'.format(self.labels[int(class_numbers[i])], confidences[i])

                cv2.putText(imagem, text_box_current, (x_min, y_min - 7), cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, colour_box_current, 5)

        if title is not None:
            plt.title(title)  #

        return imagem

    def processImage(self, path, probability_minimum=None, threshold=None, title=None, show_number=False):

        if probability_minimum != None:
            self.probability_minimum = probability_minimum

        if threshold != None:
            self.threshold = threshold

        image, image_data, image_type, resized_image = self.pre_process_image(path, model_image_size=(416, 416))
        # imagem = cv2.imread(path)
        shape = image.shape
        h, w = shape[:2]

        tensor = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        self.network.setInput(tensor)
        output = self.network.forward(self.layers_names_output)

        bounding_boxes, confidences, class_numbers, bounding_boxes2 = self.detect(output, h, w)

        sorted_indices = sorted(range(len(confidences)), key=lambda i: confidences[i], reverse=True)
        select_indices = self.non_max_suppression(bounding_boxes2, confidences)

        imagem = self.getNewImage(image, bounding_boxes, class_numbers, confidences, select_indices, title)

        number = len(select_indices)
        
        count_objects = []

        for indice in select_indices:
            count_objects.append(class_numbers[indice])

        return imagem, number, count_objects
