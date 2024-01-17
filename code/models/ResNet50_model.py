import numpy as np
import json
from PIL import Image
import io
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

from .abstract_model import AbstractModel

class ResNet50Model(AbstractModel):

	def get_category(self, imagen_bytes):
		data = {"success": False}
    
		# Load the model
		local_weights_path = 'resnet50_weights_tf_dim_ordering_tf_kernels.h5'
		model = ResNet50(weights=local_weights_path)

		imagen = Image.open(io.BytesIO(imagen_bytes))
		img = imagen.resize((224, 224))
		#img = image.load_img(img_path, target_size=(224, 224))

		# turn the image into a numpy array
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)

		# run the inference
		preds = model.predict(x)

		# decode the results into a list of tuples (class, description, probability)
		# (one such list for each sample in the batch)
		results = decode_predictions(preds, top=3)[0]

		data["predictions"] = []

		print(results)
		# loop over the results and add them to the list of
		# returned predictions
		for (imagenetID, label, prob) in results:
			r = {"label": label, "probability": float(prob)}
			data["predictions"].append(r)

		# indicate that the request was a success
		data["success"] = True
  
  		# indica el modelo que ha sido utilizado
		data["model"] = 'ResNet50'
		
		return data







