from model import resolve_single
from model.edsr import edsr

from utils import load_image, plot_sample

model = edsr(scale=4, num_res_blocks=16)
model.load_weights('weights/edsr-16-x4/weights.h5')

lr = load_image('demo/frame-1214_frame_blend')
sr = resolve_single(model, lr)

plot_sample(lr, sr)
