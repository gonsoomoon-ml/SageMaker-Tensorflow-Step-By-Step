import os
import tensorflow as tf

HEIGHT = 32
WIDTH = 32
DEPTH = 3
NUM_CLASSES = 10
NUM_DATA_BATCHES = 5
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 10000 * NUM_DATA_BATCHES

classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

def get_filenames(channel_name, channel):
    if channel_name in ['train', 'validation', 'eval']:
        return [os.path.join(channel, channel_name + '.tfrecords')]
    else:
        raise ValueError('Invalid data subset "%s"' % channel_name)


        
def _dataset_parser(value):
    """Parse a CIFAR-10 record from value."""
    featdef = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }

    example = tf.io.parse_single_example(value, featdef)
    image = tf.io.decode_raw(example['image'], tf.uint8)
    image.set_shape([DEPTH * HEIGHT * WIDTH])

    image = tf.cast(
        tf.transpose(tf.reshape(image, [DEPTH, HEIGHT, WIDTH]), [1, 2, 0]),
        tf.int32)

    label = tf.cast(example['label'], tf.int32)

    return image, tf.one_hot(label, NUM_CLASSES)

def _input(epochs, batch_size, channel, channel_name):
    filenames = get_filenames(channel_name, channel)
    dataset = tf.data.TFRecordDataset(filenames)

    # TFRecord 갯수 세기
    ds_size = sum(1 for _ in dataset)  

    
    # 데이터 세트 파싱
    dataset = dataset.map(_dataset_parser, num_parallel_calls=10)
    dataset = dataset.repeat(1)    # 1번 반복
    
    # 후련 데이터의 경우에는 셔블링을 전체 레코드 사이즈로 함.
    if channel_name == 'train':
        buffer_size = ds_size # 셔블리의 사이즈를 전체 레코드 갯수로 제공 함.
        dataset = dataset.shuffle(buffer_size=buffer_size)
 
    
    # 데이터 세트를 배치 사이즈로 나누고, 캐싱
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE) # 캐싱 함

    return dataset

