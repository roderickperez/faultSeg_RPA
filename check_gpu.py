import tensorflow as tf
import torch

print("TensorFlow GPU devices:", tf.config.list_physical_devices('GPU'))
print("PyTorch CUDA available:", torch.cuda.is_available())
