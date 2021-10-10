import os


import tensorflow as tf
tf.random.set_seed(42)

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

(mnist_images, mnist_labels), _ = tf.keras.datasets.mnist.load_data(
    path="mnist-%d.npz" % 10
)

print("type: ", type(mnist_images))
print("shape: ", mnist_images.shape)
print("mnist_labels: ", mnist_labels.shape)

dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(mnist_images[..., tf.newaxis] / 255.0, tf.float32), 
     tf.cast(mnist_labels, tf.int64))
)

batch_size = 256
dataset = dataset.repeat().shuffle(10000).batch(batch_size)

mnist_model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(32, [3, 3], activation="relu"),
        tf.keras.layers.Conv2D(64, [3, 3], activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)

loss = tf.losses.SparseCategoricalCrossentropy()

opt = tf.optimizers.Adam(0.000125 * 8)

checkpoint = tf.train.Checkpoint(model=mnist_model, optimizer=opt)


@tf.function
def training_step(images, labels, first_batch):
    with tf.GradientTape() as tape:
        probs = mnist_model(images, training=True)
        loss_value = loss(labels, probs)
        
    grads = tape.gradient(loss_value, mnist_model.trainable_variables)
    opt.apply_gradients(zip(grads, mnist_model.trainable_variables))
    
    return loss_value


#for epoch in range(EPOCHS):
n_batch = 10000
print_interval = 1000
for batch, (images, labels) in enumerate(dataset.take(n_batch)):
    loss_value = training_step(images, labels, batch == 0)

    if batch % print_interval == 0:
        print("Step #%d\tLoss: %.6f" %  (batch, loss_value))
        
checkpoint_dir = os.environ["SM_MODEL_DIR"]

mnist_model.save(os.path.join(checkpoint_dir, "1"))