from utils import load_image, plot_sample, save_image
from model.srgan import generator
from model import resolve_single

model = generator()
model.load_weights('weights/srgan/gan_generator.h5')

lr = load_image('demo/frame_3.png')
sr = resolve_single(model, lr)

save_image("test.png",sr)
#plot_sample(lr, sr)
