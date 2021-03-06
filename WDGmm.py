import numpy as np
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture as GMM


class WDGmm:
	def __init__(self):
		self.gmm_model = None
	

	def train_gmm(self, case, prp):
	  ref_img = plt.imread(prp.samples_dict[case].reference, 0)
	  orig_shape = ref_img.shape

	  ref_image_flat = np.reshape(ref_img, (-1, 1))
	  self.gmm_model = GMM(n_components=3, covariance_type='diag').fit(np.reshape(ref_image_flat, (-1, 1)))
	 

	def predict_gmm(self, case, prp, is_ref=True):
	  assert(self.gmm_model is not None)

	  img = plt.imread(prp.samples_dict[case].reference, 0) if is_ref \
		else plt.imread(prp.samples_dict[case].inspected, 0)

	  orig_shape = img.shape
	  img_flat = np.reshape(img, (-1, 1))

	  gmm_predict_labels = self.gmm_model.predict(img_flat)
	  gmm_predict_probs = self.gmm_model.predict_proba(img_flat)

	  probs_image = np.array([gmm_predict_probs[i,gmm_predict_labels[i]] for i in range(img_flat.size)])

	  #print(gmm_predict_labels.shape)
	  #print(probs_image.shape)

	  gmm_predict_labels = np.reshape(gmm_predict_labels, orig_shape)
	  probs_image = np.reshape(probs_image, orig_shape)

	  return(gmm_predict_labels, probs_image)

