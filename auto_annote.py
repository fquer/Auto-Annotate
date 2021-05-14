"""

! marks are required

trained model required

fill blanks with your paths and just run the py script

"""

import cv2
import numpy as np
import tensorflow as tf
import os
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from math import ceil

""" PATHS must end with \\ """

IMAGE_PATH = '' # images folder path !
SAMPLE_PATH = '' # samples folder path !
XML_PATH = '' # xmls folder path !
CONFIG_PATH = '' # pipeline config file path !

# does not end with \\
CHECKPOINT_PATH = '' # model checkpoint folder path !

labels={""} #include class names here !

label_id_offset = 1 #number of started jpg files ex: hello_1 -> hello_128 !
splitter = '_' #ex : hello-1.jpg  splitter is -    !
max_id_label = 100 #this number not included   !

# XML Settings

FOLDER = ''  #any folder name !
WIDTH = ''  #jpg width and height !
HEIGHT = '' # !

def generate_xml(LABEL,ID):
    try:
        fin = open(SAMPLE_PATH + "sample.xml","rt")

        fout = open(XML_PATH + "{}{}{}.xml".format(LABEL, splitter, str(ID)),"wt")

        data = fin.read()
        fout.write(data)

        fin.close()
        fout.close()
    except:
        print("Generate Passed")

def edit_xml(LABEL,ID,XMIN,YMIN,XMAX,YMAX):
    try:
        
        PATH = IMAGE_PATH + '{}{}{}.jpg'.format(LABEL, splitter, str(ID))  # any path of jpg file

        fin = open(XML_PATH + "{}{}{}.xml".format(LABEL, splitter, str(ID)),"rt")

        data = fin.read()

        fin.close()

        data = data.replace('FOLDER', FOLDER)
        data = data.replace('FILENAME', LABEL + '_' + str(ID) + '.jpg')
        data = data.replace('PATH', PATH)
        data = data.replace('WIDTH', WIDTH)
        data = data.replace('HEIGHT', HEIGHT)
        data = data.replace('LABEL', LABEL)
        data = data.replace('XMIN', str(XMIN))
        data = data.replace('YMIN', str(YMIN))
        data = data.replace('XMAX', str(XMAX))
        data = data.replace('YMAX', str(YMAX))

        fout = open(XML_PATH + "{}{}{}.xml".format(LABEL, splitter, str(ID)),"wt")

        fout.write(data)

        fout.close()
    except:
        print("Edit Passed")

def expand_xml(LABEL,ID):
    try:
        fin = open(XML_PATH + "{}{}{}.xml".format(LABEL, splitter, str(ID)),"rt")

        fnewclass = open(SAMPLE_PATH + "new_class_sample.xml","rt")

        new_class=fnewclass.readlines()
        contents=fin.readlines()
        new_contents=[]

        fin.close()

        fout = open(XML_PATH + "{}{}{}.xml".format(LABEL, splitter, str(ID)),"wt")

        count = 0
        for content in contents:

            if content == "	</object>\n" and count == 0:
                new_contents.append(content)
                count=1
                for lines in new_class:
                    new_contents.append(lines)

            else:
                new_contents.append(content)

        for i in new_contents:
            fout.write(i)


        fnewclass.close()
        fout.close()
    except:
        print("Expand Passed")

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)


ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-0')).expect_partial() #select your checkpoint !

category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH+'\\label_map.pbtxt') #include label map here !


for label in labels:
    for id in range(label_id_offset,max_id_label):

        try:
            frame = cv2.imread(IMAGE_PATH + '{}{}{}.jpg'.format(label, splitter,str(id)))
            image_np = np.array(frame)

            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)

            image, shapes = detection_model.preprocess(input_tensor)
            prediction_dict = detection_model.predict(image, shapes)
            detections = detection_model.postprocess(prediction_dict, shapes)

            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                          for key, value in detections.items()}
            detections['num_detections'] = num_detections


            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

            id_offset = 1
            image_np_with_detections = image_np.copy()

            viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes'] + id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=1,
                min_score_thresh=.5,
                agnostic_mode=False)


            coordinates = viz_utils.return_coordinates(
                frame,
                np.squeeze(detections['detection_boxes']),
                np.squeeze(detections['detection_classes']).astype(np.int32) + id_offset,
                np.squeeze(detections['detection_scores']),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8,
                min_score_thresh=0.70) #min score thresh to annote !


            if len(coordinates)==0:
                print("Generate XML Passed : Detection Failed")

            else:
                generate_xml(label,id)

                for count in range(0,len(coordinates)):

                    if count > 0 :
                        expand_xml(label,id)

                    ymin = coordinates[count][0]
                    ymax = coordinates[count][1]
                    xmin = coordinates[count][2]
                    xmax = coordinates[count][3]
                    acc  = coordinates[count][4]
                    name = coordinates[count][5]

                    print("{}{}{}.jpg coordinate values".format(label,splitter, str(id)))
                    print("xmin : ", xmin," ymin : ", ymin," xmax : ", xmax," ymax : ", ymax," acc : ",ceil(acc)," class : ",name,"\n\n")

                    edit_xml(name, id, xmin, ymin, xmax, ymax)

        except:

            print("Detect Passed\n\n")

        # if you want you can see detections

        #cv2.imshow('{}_{}.jpg'.format(label,str(id)), cv2.resize(image_np_with_detections, (800, 600)))
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
