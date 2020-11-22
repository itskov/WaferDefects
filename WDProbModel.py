import numpy as np

class WDProbModel:
	def __init__(self, labels_count):
		self.labels = range(labels_count)
		self.pattern_dict = {}
		self.pattern_dict_count = {}

		self.count = 0


	def add_data(self, img_ref, img_insp):
		assert(np.all(img_ref.shape == img_insp.shape))

		#print(img_ref)

		# Going over the matrix and collecting data
		for i in range(1, img_ref.shape[0] -1):
			for j in range(1, img_ref.shape[1] - 1):
				window =  img_ref[(i-1):(i+2), (j-1):(j+2)]
				#print(window)
				ref_key = tuple([np.sum(window == val) for val in self.labels])
				insp_ley = img_insp[i - 1, j - 1]
				key = (ref_key, insp_ley)
				
				self.pattern_dict[key] = (self.pattern_dict[key] + 1) if key in self.pattern_dict else 1

				self.pattern_dict_count[ref_key] = (self.pattern_dict_count[ref_key] + 1) \
									if ref_key in self.pattern_dict_count else 1.0

				#print(self.pattern_dict)
				#input("Press Enter to continue...")
		
		self.count += np.sum(list(self.pattern_dict.values()))
		#print(self.count)

	
	def get_probability(self, key):
		return (self.pattern_dict[key] / self.pattern_dict_count[key[0]]) if key in self.pattern_dict \
			else 0
	
	def get_image_probability(self, img_ref, img_insp):
		assert(np.all(img_ref.shape == img_insp.shape))
		
		res = np.zeros(np.array(img_ref.shape) - 2)
		for i in range(1, img_ref.shape[0] -1):
			for j in range(1, img_ref.shape[1] - 1):
				window =  img_ref[(i-1):(i+2), (j-1):(j+2)]
				ref_key = tuple([np.sum(window == val) for val in self.labels])
				insp_ley = img_insp[i - 1, j - 1]
				key = (ref_key, insp_ley)
				res[i - 1, j - 1] = self.get_probability(key)

		return res;



if __name__ == "__main__":
	a = WDProbModel(3)
	np.random.seed(1)
	b = np.random.randint(0, 3, (20,20))
	a.add_data(b, b)

	print(a.pattern_dict)

	print(a.get_probability(((2,3,4),0)))
	print(a.get_probability(((2,3,4),1)))
	print(a.get_probability(((2,3,4),2)))
		


		
