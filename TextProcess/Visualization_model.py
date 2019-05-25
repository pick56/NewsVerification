from keras.utils import plot_model
from keras.models import load_model

model = load_model(r'model_save/news_social_bilstm_model3.h5')
plot_model(model, show_shapes=True, to_file='model6.png')
