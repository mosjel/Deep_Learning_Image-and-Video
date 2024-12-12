import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import tensorflow as tf
import numpy as np
import cv2
import pathlib

def load_data(address):
    img=cv2.imread(address)
    print(img.shape)
    r_img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    r_img=np.expand_dims(r_img,axis=0)
    print(r_img.shape)
    r_img=tf.convert_to_tensor(r_img)

    return(img,r_img)


def load_model():
    model_dir="Your Model Path"
    model_dir=pathlib.path(model_dir)/"saved_model"
    detection_model=tf.saved_model.load(str(model_dir))
    return (detection_model)


def inference(img,model):
    output_dict=model(img)
    num_detections=int(output_dict.pop("num_detections"))
    output_dict={key:value[0,:num_detections].numpy() for key,value in output_dict.items()}
    output_dict["detection_classes"]=np.array(output_dict["detection_classes"]).astype(np.int64)
    return(output_dict)

def visualize(output_dict,img):
    path_to_labels="your file name.pbtxt"
    category_index=label_map_util.create_categories_from_labelmap(path_to_labels,use_display_name=True)
    vis_util.visualize_boxes_and_labels_on_image_array(img,
                                                       output_dict["detection_boxes"],
                                                       output_dict["detection_classes"],
                                                       output_dict["detection_scores"],
                                                       category_index,
                                                       use_normalized_coordinates=True 
                                                       )
    cv2.imshow("img",img)
    cv2.waitKey(0)


img,r_img=load_data("image1.jpg")
detection_model=load_model()
output_dict=inference(r_img,detection_model)
visualize(output_dict,img)



