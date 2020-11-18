from os import path, mkdir
from glob import glob
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np

import json
import shutil
import cv2

"""
This class governs the preprocessing 
of the given data.
"""
class WDPreprocessing:
  def __init__(self, defects_dir, no_defects_dir):
    ALIGNED_FRAME_DIR = './aligned_frame_data'
    self.samples_dict = {}

    # Gathering sample data.
    Sample = namedtuple('Sample',['inspected', 'reference'])
  
    # Going over the defective files.
    defects_files = glob(path.join(defects_dir, "case*.tif"))
    assert(len(defects_files) == 4)
    
    # Going over the non defective files.
    no_defects_files = glob(path.join(no_defects_dir, "case*.tif"))
    assert(len(no_defects_files) == 2)

    # getting the unique names of samples we have.
    all_files = np.array(defects_files + no_defects_files)
    unique_cases = np.unique([path.basename(f).split("_")[0] for f in all_files])

    # Arranging the data a bit.
    for case in unique_cases:
      current_files =  np.array(all_files[[(case in c) for c in all_files]])
      assert(len(current_files) == 2)

      ref = current_files[[('reference' in f) for f in current_files]]
      ins = current_files[[('inspected' in f) for f in current_files]]
    
      self.samples_dict[case] = Sample(ins, ref)

    # Reading the defects txt file.
    defects_file = path.join(defects_dir,'defects.json')
    assert(path.exists(defects_file))

    # Reading the defects txt file.
    with open(defects_file) as f:
      self.defects_dict = json.load(f)

    # Orginizing the full-frame data.
    if path.exists(ALIGNED_FRAME_DIR):
      shutil.rmtree(ALIGNED_FRAME_DIR)

    mkdir(ALIGNED_FRAME_DIR)

    for case in unique_cases:
      sample_dir = path.join(ALIGNED_FRAME_DIR, case)
      if not path.exists(sample_dir):
        mkdir(sample_dir)


      print(self.samples_dict[case].inspected)
      inspected_image = cv2.imread(self.samples_dict[case].inspected[0])
      reference_image = cv2.imread(self.samples_dict[case].reference[0])

      # Converting to greyscale
      inspected_image = cv2.cvtColor(inspected_image, cv2.COLOR_BGR2GRAY)
      reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

      # aligning inspected to reference
      print(inspected_image.shape)
      aligned_inspected, aligned_reference = \
            WDPreprocessing.align_images(inspected_image, reference_image, 
                                         range(100,200), range(50, 250))
            
      cv2.imwrite(path.join(sample_dir,'inspected.tif'), inspected_image)
      cv2.imwrite(path.join(sample_dir,'reference.tif'), reference_image)

  '''
  This method aligns an image to its reference using a kernel on 
  the reference image.
  '''
  @staticmethod
  def align_images(inspected, reference, range_kernel_y, range_kernel_x, verbose=False, plot=False):
    kernel = reference[np.min(range_kernel_y):np.max(range_kernel_y),
                       np.min(range_kernel_x):np.max(range_kernel_x)]

    # Using cross-correlation for pattern matching.
    res = cv2.matchTemplate(inspected, kernel,cv2.TM_CCOEFF)
    min_val, max_val, min_loc, top_left = cv2.minMaxLoc(res)
  

    orig_top_left = np.array((np.min(range_kernel_y), np.min(range_kernel_x)))
    cross_cor_top_lef = np.array((top_left[1], top_left[0]))

    # Calculating te shift.
    shift = cross_cor_top_lef - orig_top_left
    print(shift)

    if verbose:
      print('Orig top left: %d, %d' % (orig_top_left[0], orig_top_left[1]))
      print('Cross-corr top left: %d %d ' % (top_left[1], top_left[0]))

    #print((top_left[0] + kernel.shape[1], top_left[1] + kernel.shape[0]))
    bottom_right = (top_left[0] + kernel.shape[1], top_left[1] + kernel.shape[0])


    new_inspected = inspected.copy()
    new_reference = reference.copy()

    if np.sign(shift[0]) == 1:
      new_inspected = new_inspected[shift[0]:, :]
      new_reference = new_reference[:-shift[0], :]
    else:
      new_inspected = new_inspected[:shift[0], :]
      new_reference = new_reference[-shift[0]:, :]

    if np.sign(shift[1]) == 1:
      new_inspected = new_inspected[:, shift[1]:]
      new_reference = new_reference[:, :-shift[1]]
    else:
      new_inspected = new_inspected[:, :shift[1]]
      new_reference = new_reference[:, -shift[1]:]

    if plot:
      inspected_rec = inspected.copy()
      cv2.rectangle(inspected_rec,top_left, bottom_right, 0, 6)
      cv2.rectangle(inspected_rec,
                    (np.min(range_kernel_x),np.min(range_kernel_y)),
                    (np.max(range_kernel_x),np.max(range_kernel_y)), 255, 6)
      fig = plt.figure()
      ax1=plt.subplot(1, 3, 1)
      ax1.imshow(inspected_rec)
      plt.title('Cross Correlation')
      ax2=plt.subplot(1, 3, 2)
      ax2.imshow(new_reference)
      plt.title('Reference')
      ax3 = plt.subplot(1, 3, 3)
      ax3.imshow(new_inspected)
      plt.title('Inspected')

      fig.tight_layout()
      
    return(new_inspected, new_reference)



      

    

#plt.imshow(plt.imread('/content/full_frame_data/case2/inspected.tif'))

"""
plt.figure()
plt.imshow(plt.imread('/content/full_frame_data/case1/inspected.tif'))
defect = prp.defects_dict['case1'][0]
plt.scatter(defect[0], defect[1])
defect = prp.defects_dict['case1'][1]
plt.scatter(defect[0], defect[1])
defect = prp.defects_dict['case1'][2]
plt.scatter(defect[0], defect[1])
"""

#im1 = plt.imread('/content/full_frame_data/case2/inspected.tif')
#im2 = plt.imread('/content/full_frame_data/case2/reference.tif')[100:250, 150:400]
#im2 = plt.imread('/content/full_frame_data/case2/reference.tif')



#im1 = im1[:,:]
#im2 = im2[:,:]

#plt.figure()
#plt.imshow(im1[:, :])
#plt.figure()
#plt.imshow(im2[:,:])

#im2_b =  cv2.adaptiveThreshold(im2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,7,7)
#plt.figure()
#plt.imshow(im2_b)

"""
result = []
methods = [cv2.TM_CCOEFF]
for i in range(len(methods)):
    res = cv2.matchTemplate(im1,im2_b,methods[i])
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    print(top_left)
    print((top_left[0] + im2.shape[1], top_left[1] + im2.shape[0]))
    bottom_right = (top_left[0] + im2.shape[1], top_left[1] + im2.shape[0])

    im1_rec = im1.copy()
    cv2.rectangle(im1_rec,top_left, bottom_right, 255, 2)
    cv2.rectangle(im1_rec,(150, 100), (400,250), 20, 3)
    plt.figure()
    plt.imshow(im1_rec)

im2 = plt.imread('/content/full_frame_data/case2/reference.tif')
plt.figure()
plt.imshow(im1[:-6,:-5])
plt.figure()
plt.imshow(im2[6:,5:])
"""

    #print ("Method {}  : Result{}") .format(method[i],res)


