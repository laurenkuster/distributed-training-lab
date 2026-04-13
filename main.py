
import time
import tensorflow as tf
import mnist

batch_size = 64

dataset = mnist.mnist_dataset(batch_size)
model = mnist.build_and_compile_cnn_model()

print("\n=== RUNNING TRAINING (SIMULATED MULTI-WORKER SETUP) ===")

start_time = time.time()

history = model.fit(
    dataset,
    epochs=3,
    steps_per_epoch=70
)

end_time = time.time()

print("\n=== TRAINING SUMMARY ===")
print(f"Total training time: {end_time - start_time:.2f} seconds")
print(f"Final accuracy: {history.history['accuracy'][-1]:.4f}")
