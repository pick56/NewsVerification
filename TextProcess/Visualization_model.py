from keras.utils import plot_model
from keras.models import load_model

"""
读取模型，并结构图
"""
model = load_model(r'model_save/news_social_bilstm_model3.h5')
plot_model(model, show_shapes=True, to_file='model6.png')
