import os


import tensorflow as tf
tf.random.set_seed(42)

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

####################################
# DDP 
####################################

# Import SMDataParallel TensorFlow2 Modules
import smdistributed.dataparallel.tensorflow as dist


# SMDataParallel: Initialize
dist.init()

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    # SMDataParallel: Pin GPUs to a single SMDataParallel process [use SMDataParallel local_rank() API]
    tf.config.experimental.set_visible_devices(gpus[dist.local_rank()], "GPU")

####################################


print("dist.rank(): ", dist.rank())
print("mnist files: mnist-%d.npz" % dist.rank())

####################################
# DDP 
####################################
(mnist_images, mnist_labels), _ = tf.keras.datasets.mnist.load_data(
    path="mnist-%d.npz" % dist.rank()
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

# opt = tf.optimizers.Adam(0.000125 * 8)
opt = tf.optimizers.Adam(0.000125 * dist.size()) # DDP

checkpoint = tf.train.Checkpoint(model=mnist_model, optimizer=opt)


@tf.function
def training_step(images, labels, first_batch):
    with tf.GradientTape() as tape:
        probs = mnist_model(images, training=True)
        loss_value = loss(labels, probs)

    # DDP    
    # SMDataParallel: Wrap tf.GradientTape with SMDataParallel's DistributedGradientTape
    tape = dist.DistributedGradientTape(tape)

        
    grads = tape.gradient(loss_value, mnist_model.trainable_variables)    
    opt.apply_gradients(zip(grads, mnist_model.trainable_variables))

    # DDP
    if first_batch:
        # SMDataParallel: Broadcast model and optimizer variables
        dist.broadcast_variables(mnist_model.variables, root_rank=0)
        dist.broadcast_variables(opt.variables(), root_rank=0)

    # SMDataParallel: all_reduce call
    loss_value = dist.oob_allreduce(loss_value)  # Average the loss across workers
    
    return loss_value


#for epoch in range(EPOCHS):
n_batch = 10000
print_interval = 1000
for batch, (images, labels) in enumerate(dataset.take(n_batch)):
    loss_value = training_step(images, labels, batch == 0)

#     if batch % print_interval == 0:    
    if batch % print_interval == 0 and dist.rank() == 0:    
        print("Step #%d\tLoss: %.6f" %  (batch, loss_value))
        


# SMDataParallel: Save checkpoints only from master node.
if dist.rank() == 0:
    checkpoint_dir = os.environ["SM_MODEL_DIR"]
    mnist_model.save(os.path.join(checkpoint_dir, "1"))