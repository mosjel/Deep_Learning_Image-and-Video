from absl import logging

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from scipy.spatial import cKDTree
from skimage.feature import plot_matches
from skimage.measure import ransac
from skimage.transform import AffineTransform
from six import BytesIO

import tensorflow as tf
import glob

import tensorflow_hub as hub
from six.moves.urllib.request import urlopen
def download_and_resize(name, url, new_width=256, new_height=256):
    # path = tf.keras.utils.get_file(url.split('/')[-1], url)
    image = Image.open(url)
    if image is None:
        print ("*****************------------************")
        exit()
    image = ImageOps.fit(image, (new_width, new_height), Image.LANCZOS)
    return image
# IMAGE_1_URL = 'https://images.freeimages.com/images/large-previews/8ca/peerless-chain-1-1641825.jpg'
# IMAGE_2_URL = 'https://upload.wikimedia.org/wikipedia/commons/3/3e/GoldenGateBridge.jpg'
# plt.subplot(1,2,1)
# plt.imshow(image1)
# plt.subplot(1,2,2)
# plt.imshow(image2)
# plt.show()

delf = hub.load('https://tfhub.dev/google/delf/1').signatures['default']
def run_delf(image):
    np_image = np.array(image)
    float_image = tf.image.convert_image_dtype(np_image, tf.float32)


    return delf(
      image=float_image,
      score_threshold=tf.constant(100.0),
      image_scales=tf.constant([0.25, 0.3536, 0.5, 0.7071, 1.0, 1.4142, 2.0]),
      max_feature_num=tf.constant(1000))
#@title TensorFlow is not needed for this post-processing and visualization
def match_images(image1, image2, result1, result2):
  distance_threshold = 0.8

  # Read features.
  num_features_1 = result1['locations'].shape[0]
  print("Loaded image 1's %d features" % num_features_1)
  
  num_features_2 = result2['locations'].shape[0]
  print("Loaded image 2's %d features" % num_features_2)

  # Find nearest-neighbor matches using a KD tree.
  d1_tree = cKDTree(result1['descriptors'])
  _, indices = d1_tree.query(
      result2['descriptors'],
      distance_upper_bound=distance_threshold)

  # Select feature locations for putative matches.
  locations_2_to_use = np.array([
      result2['locations'][i,]
      for i in range(num_features_2)
      if indices[i] != num_features_1
  ])
  locations_1_to_use = np.array([
      result1['locations'][indices[i],]
      for i in range(num_features_2)
      if indices[i] != num_features_1
  ])
  print(locations_1_to_use.shape,locations_2_to_use.shape)
  if locations_2_to_use.shape[0]<4:
        inliers1=0
  else:  
    
    # Perform geometric verification using RANSAC.
    _, inliers = ransac(
        (locations_1_to_use, locations_2_to_use),
        AffineTransform,
        min_samples=3,
        residual_threshold=20,
        max_trials=1000)
    if inliers is not None :
        print(type (inliers))
        print('Found %d inliers' % sum(inliers))
        inliers1=sum(inliers)
    else: inliers1=0  
  return (inliers1)

  # Visualize correspondences.
#   _, ax = plt.subplots()
#   inlier_idxs = np.nonzero(inliers)[0]
#   plot_matches(
#       ax,
#       image1,
#       image2,
#       locations_1_to_use,
#       locations_2_to_use,
#       np.column_stack((inlier_idxs, inlier_idxs)),
#       matches_color='b')
#   ax.axis('off')
#   ax.set_title('DELF correspondences')
#   plt.show()

def add_value(dict_obj, key, value):
    if key not in dict_obj or dict_obj.get(key)<value:
        dict_obj[key] = value

IMAGE_1_URL='fire.25.jpg'
image1 = download_and_resize('image_1.jpg', IMAGE_1_URL)

result1 = run_delf(image1)

res={}
for item in glob.glob(r'test_fire\*'):


    IMAGE_2_URL=item
    res_key=item.split('\\')[-1]
    image2 = download_and_resize('image_2.jpg', IMAGE_2_URL)
    result2 = run_delf(image2)
    print(res_key)
    res_val=match_images(image1, image2, result1, result2)
    add_value(res,res_key,res_val)
res = sorted([(v, k) for (k, v) in res.items()],reverse=True)
print(res[:10])


