import os

from models.ResNet50_model import ResNet50Model
from models.VGG16_model import VGG16Model
from models.InceptionResNetV2_model import InceptionResNetV2Model
from models.NASNetLarge_model import NASNetLargeModel

def get_category(imagen, tipo_modelo):
    if tipo_modelo == 'resnet50':
        model = ResNet50Model() #OK
        pass
    elif tipo_modelo == 'vgg16':
        model = VGG16Model() #OK
        pass
    elif tipo_modelo == 'inception':
        model = InceptionResNetV2Model() #OK
    elif tipo_modelo == 'nasnetlarge':
        model = NASNetLargeModel() #OK
    else:
        model = ResNet50Model() #OK
        pass

    data = model.get_category(imagen)

    return data