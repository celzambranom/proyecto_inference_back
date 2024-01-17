import numpy as np
from PIL import Image
import io
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions

from .abstract_model import AbstractModel

class InceptionResNetV2Model(AbstractModel):

	def get_category(self, imagen_bytes):
		data = {"success": False}
    
		# Load the model
		#model = InceptionResNetV2(weights='imagenet')
		local_weights_path = 'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5'
		model = InceptionResNetV2(weights=local_weights_path)

		imagen = Image.open(io.BytesIO(imagen_bytes))
		img = imagen.resize((299, 299))
		#img = image.load_img(img_path, target_size=(299, 299))

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
		data["model"] = 'InceptionResNetV2'
		
		return data







