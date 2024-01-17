import numpy as np
from PIL import Image
import io
from tensorflow.keras.applications.nasnet import NASNetLarge
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.nasnet import preprocess_input, decode_predictions

from .abstract_model import AbstractModel

class NASNetLargeModel(AbstractModel):

	def get_category(self, imagen_bytes):
		data = {"success": False}
    
		# Load the model
		#model = NASNetLarge(weights='imagenet')
		local_weights_path = 'nasnet_large.h5'
		model = NASNetLarge(weights=local_weights_path)

		imagen = Image.open(io.BytesIO(imagen_bytes))
		img = imagen.resize((331, 331))
		#img = image.load_img(imagen, target_size=(331, 331))

		# turn the image into a numpy array
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)

		# run the inference
		preds = model.predict(x)
		
		# Decode predictions
		results = decode_predictions(preds, top=3)[0]
		print('Predicted:', results)
		
		data["predictions"] = []

		# loop over the results and add them to the list of
		# returned predictions
		for (imagenetID, label, prob) in results:
			r = {"label": label, "probability": float(prob)}
			data["predictions"].append(r)
			
		# indicate that the request was a success
		data["success"] = True
  
  		# indica el modelo que ha sido utilizado
		data["model"] = 'NASNetLarge'
		
		return data







