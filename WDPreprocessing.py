from os import path, mkdir, makedirs
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

      inspected_image = cv2.imread(self.samples_dict[case].inspected[0])
      reference_image = cv2.imread(self.samples_dict[case].reference[0])

      # Converting to greyscale
      inspected_image = cv2.cvtColor(inspected_image, cv2.COLOR_BGR2GRAY)
      reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

      # aligning inspected to reference
      aligned_inspected, aligned_reference, shift = \
            WDPreprocessing.align_images(inspected_image, reference_image, 
                                         range(100,200), range(50, 250))
            
      if case in self.defects_dict:
        # Updating the defects according to the shift.
        self.defects_dict[case] = \
            [(dx[0] - shift[0], dx[1] - shift[1]) for dx in self.defects_dict[case]]
            
      new_ins_file = path.join(sample_dir,'inspected.tif')
      new_ref_file = path.join(sample_dir,'reference.tif')
      cv2.imwrite(new_ins_file, aligned_inspected)
      cv2.imwrite(new_ref_file, aligned_reference)

      # Correcting the positions to the aligned directory
      self.samples_dict[case] = Sample(new_ins_file, new_ref_file)


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

    return(new_inspected, new_reference, list(shift))


  def sample_duo_from_case(self, case, dim, count, out_dir):
        filename_ref = self.samples_dict[case].reference 
        filename_ins = self.samples_dict[case].inspected

        current_image_ref = cv2.imread(filename_ref, 0)
        current_image_ins = cv2.imread(filename_ins, 0)

        if not path.exists(out_dir):
          makedirs(out_dir)

        for i in range(count):
          y_cord = np.random.randint(dim, current_image_ref.shape[0] - dim, 1)[0]
          x_cord = np.random.randint(dim, current_image_ref.shape[1] - dim, 1)[0]

          #new_sample = current_image[y_cord:(y_cord + dim), x_cord:(x_cord + dim)]
          new_sample_ref = self.sample_from_point(current_image_ref,
                                                  (y_cord, x_cord),
                                                  dim, rand=False)
          
          new_sample_ins = self.sample_from_point(current_image_ins,
                                                  (y_cord, x_cord),
                                                  dim, rand=False)
                    

          for j in range(0,4):
            new_sample_ref = cv2.rotate(new_sample_ref, cv2.ROTATE_90_CLOCKWISE)
            new_sample_ins = cv2.rotate(new_sample_ins, cv2.ROTATE_90_CLOCKWISE)

            record = np.concatenate((new_sample_ref[..., None], 
                                     new_sample_ins[..., None]), axis = 2)

            np.save(path.join(out_dir, "record%d_r%d.npy" % (i, j)), record)


        
  def sample_from_case(self, case, from_ref, dim, count, out_dir, aug=False):
    filename = self.samples_dict[case].reference if from_ref \
              else self.samples_dict[case].inspected
    
    current_image = cv2.imread(filename, 0)
    #current_image = self.stabilize(current_image)

    #Debug
    current_image[current_image > 175] = 255
    #Debug

    if not path.exists(out_dir):
      makedirs(out_dir)

    for i in range(count):
      y_cord = np.random.randint(dim, current_image.shape[0] - dim, 1)[0]
      x_cord = np.random.randint(dim, current_image.shape[1] - dim, 1)[0]

      #new_sample = current_image[y_cord:(y_cord + dim), x_cord:(x_cord + dim)]
      new_sample = self.sample_from_point(current_image,(y_cord, x_cord), dim)

      cv2.imwrite(path.join(out_dir, '%s_%d_r0.tif' % (case, i)), new_sample)

      if aug:
        for j in range(0,4):
          new_sample = cv2.rotate(new_sample, cv2.ROTATE_90_CLOCKWISE)
          cv2.imwrite(path.join(out_dir, 
                                '%s_%d%c_r%d.tif' % (case, i,'r' if from_ref else 'i', 90 * j)), new_sample)


  def sample_from_point(self, current_image, point, dim, rand=True):

    current_img = self.stabilize(current_image)

    if rand:
      upper_left_y = point[0] - np.random.randint(0, dim)
      upper_left_x = point[1] - np.random.randint(0, dim)
    else:
      upper_left_y = point[0]
      upper_left_x = point[1]

    print('Point', point, 'Dim', dim)
    print('Upper Left:', upper_left_y, upper_left_x)

    #print(upper_left_x, upper_left_y)

    current_image = current_image[upper_left_y:(upper_left_y + dim), 
                                  upper_left_x:(upper_left_x + dim)]

    print(current_image.shape)

    return current_image


  def stabilize(self, img):
    offset = 85

    new_img = ((np.int16(img)- np.mean(img)) / np.std(img))
    new_img = new_img / (np.max(new_img) - np.min(new_img))
    new_img *= 255

    new_img = np.minimum(new_img, 255)
    new_img = np.maximum(new_img, 0)


    return np.uint8(new_img)



#DEBUG
prp = WDPreprocessing('./WaferDefects/raw_data/defective_examples', './WaferDefects/raw_data/non_defective_examples')


