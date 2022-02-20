import os
import cv2
import glob
import argparse
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util


class Inference:
    WORKSPACE_PATH = 'Tensorflow/workspace'
    ANNOTATION_PATH = WORKSPACE_PATH+'/annotations'
    MODEL_PATH = WORKSPACE_PATH+'/models'
    CONFIG_PATH = MODEL_PATH+'/my_ssd_mobnet/pipeline.config'
    CHECKPOINT_PATH = MODEL_PATH+'/my_ssd_mobnet/checkpoints/'
    detection_model = ""
    ckpt = ""
    category_index = ""
    threshold = .3

    def __init__(self):
        # Load pipeline config and build a detection model
        configs = config_util.get_configs_from_pipeline_file(self.CONFIG_PATH)
        self.detection_model = model_builder.build(
            model_config=configs['model'], is_training=False)

        # # Restore checkpoint
        self.ckpt = tf.compat.v2.train.Checkpoint(model=self.detection_model)
        self.ckpt.restore(os.path.join(self.CHECKPOINT_PATH,
                                       'ckpt-91')).expect_partial()
        self.category_index = label_map_util.create_category_index_from_labelmap(
            self.ANNOTATION_PATH+'/label_map.pbtxt')

    def detect_fn(self, image):
        image, shapes = self.detection_model.preprocess(image)
        prediction_dict = self.detection_model.predict(image, shapes)
        detections = self.detection_model.postprocess(prediction_dict, shapes)
        return detections

    def prediction(self, image_path="example/image_000000962.jpg"):
        frame = cv2.imread(image_path)
        # frame = cv2.resize(frame, (640, 480))
        image_np = np.array(frame)

        input_tensor = tf.convert_to_tensor(
            np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = self.detect_fn(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(
            np.int64)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        detection_boxes = detections['detection_boxes']
        detection_classes = detections['detection_classes']+label_id_offset
        detection_scores = detections['detection_scores']

        txt = ""
        height, width, _ = frame.shape
        for boxes, classes, scores in zip(detection_boxes, detection_classes, detection_scores):
            if scores >= self.threshold:
                ymin, xmin, ymax, xmax = boxes
                classes = self.category_index[classes]['name']
                txt += f"{classes} {scores} {round(xmin*width)} {round(ymin*height)} {round(xmax * width)} {round(ymax*height)}\n"
                output_txt_path = "output/" + \
                    image_path.split("/")[-1].replace(".jpg", ".txt")
                with open(output_txt_path, 'w') as f:
                    f.write(txt)
                # print([round(xmin*width), round(ymin*height),
                #        round(xmax * height), round(ymax*height)], classes, scores)
                # print(boxes, classes, scores, height, width)

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes']+label_id_offset,
            detections['detection_scores'],
            self.category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=10,
            min_score_thresh=self.threshold,
            agnostic_mode=False)

        cv2.imwrite("output/" + image_path.split("/")
                    [-1], image_np_with_detections)


if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description='Path of the image')

    # Add the arguments
    parser.add_argument('--image_path',
                        metavar='image_path',
                        type=str,
                        default="example/image_000000962.jpg",
                        help='the path to the image')

    args = parser.parse_args()
    image_path = args.image_path

    object = Inference()

    is_dir = False
    if os.path.isdir(image_path):
        is_dir = True
    elif os.path.isfile(image_path):
        is_dir = False

    if is_dir:
        image_list = glob.glob(image_path + "*.jpg")
        for img in image_list:
            object.prediction(img)
    else:
        object.prediction(image_path)

# python inference.py --image_path example/image_000000962.jpg
