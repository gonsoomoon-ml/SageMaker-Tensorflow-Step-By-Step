# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# 참고: https://github.com/horovod/horovod/blob/master/examples/tensorflow2/tensorflow2_mnist.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os

import tensorflow as tf
print("tensorflow version: ", tf.__version__)

# from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
tf.get_logger().setLevel('INFO')

from tf_model_def import tf_model_fn # 사용자 정의 모델 정의

HEIGHT = 32
WIDTH = 32
DEPTH = 3
NUM_CLASSES = 10
    

def main(args):

    #############################
    # Horovod 세팅
    #############################    

    import horovod.tensorflow as hvd
    # Horovod: initialize Horovod.
    hvd.init()

    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    

    #############################
    # 커맨드 인자 확인
    #############################    
    if hvd.local_rank() == 0:    
        print(f"## args: \n", args)    

    #############################
    # 입력 데이터 세트 준비
    #############################        

    train_dataset = train_input_fn(hvd)
    train_batch_size = sum(1 for _ in train_dataset)    

    eval_dataset = eval_input_fn(hvd)
    eval_batch_size = sum(1 for _ in eval_dataset)    
    
    validation_dataset = validation_input_fn(hvd)
    validation_batch_size = sum(1 for _ in validation_dataset)    

    
    # 각각의 데이터 세트에 대한 배치 갯수를 확인
    if hvd.local_rank() == 0: 
        print("\n################# Prepare Dataset ################")        
        print("# of batches in train: ", train_batch_size)
        print("# of batches in eval: ", eval_batch_size)                
        print("# of batches in validation: ", validation_batch_size)        
    

    #############################
    # 모델 정의: 모델 정의 가져옴.
    #############################            
    logging.info("configuring model")
    model = tf_model_fn(args.learning_rate, args.weight_decay, args.optimizer, args.momentum)
    
    #############################
    # 모델 훈련
    #############################        

    loss_object = tf.losses.SparseCategoricalCrossentropy()
    optimizer = tf.optimizers.Adam(learning_rate=args.learning_rate * hvd.size()) # Horovod 수정
    optimizer = hvd.DistributedOptimizer(optimizer)    

    

    logging.info("Starting training")    
    print("\n################# Start Training  ################")
    # 테스트 변수 정의
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


    print_interval = args.print_interval # 몇 개의 batch 에서 Loss 프린트 할지 정함
    EPOCHS = args.epochs
    
    num_train_batches = train_batch_size // hvd.size()
    print(f"## num_train_batch on each GPU{hvd.rank()} : {num_train_batches} " )

    for epoch in range(EPOCHS):
        # 모델 훈련 스넵
        for batch, (images, labels) in enumerate(train_dataset.take(num_train_batches)):        
        #for batch, (images, labels) in enumerate(train_dataset):        
            loss_value = train_step(model, loss_object, optimizer, images, labels, hvd, batch == 0)

            if batch % print_interval == 0:
                print(f"## GPU{hvd.rank()} - Step #%d\tLoss: %.6f" %  (batch, loss_value))

        if hvd.local_rank() == 0:                
            # 모델 Validation 스텝        
            test_loss.reset_states()
            test_accuracy.reset_states()

            for test_images, test_labels in validation_dataset:
                test_loss, test_accuracy = test_step(model, loss_object, test_loss, test_accuracy, test_images, test_labels)

            # Validation 출력
            template = '## Epoch {}, Test Loss: {}, Test Accuracy: {}'
            print(template.format(epoch+1,
                                test_loss.result(),
                                test_accuracy.result()*100))


    print('Training Finished.')
    print("\n################# Start Training  ################")    
    

    #############################
    # 모델 저장
    #############################            

    if hvd.local_rank() == 0:
        print("\n################# Saving Model  ################")            
        print(f"Model is saved in {args.model_output_dir}")
        save_model(model, args.model_output_dir)
        
    return None

@tf.function
def train_step(model, loss_object, optimizer, images, labels, hvd, first_batch):
    '''
    모델 훈련
    '''
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)        
        
    # Horovod: add Horovod Distributed GradientTape.
    tape = hvd.DistributedGradientTape(tape)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    #
    # Note: broadcast should be done after the first gradient step to ensure optimizer
    # initialization.
    if first_batch:
        hvd.broadcast_variables(model.variables, root_rank=0)
        hvd.broadcast_variables(optimizer.variables(), root_rank=0)    
    
    return loss


def test_step(model, loss_object, test_loss, test_accuracy, images, labels):
    '''
    검증 데이터 세트로 손실, 정확도 계산 및 기록
    '''
    predictions = model(images)
    t_loss = loss_object(labels, predictions)


    test_loss(t_loss)
    test_accuracy(labels, predictions)
    
    return test_loss, test_accuracy



def get_filenames(channel_name, channel):
    '''
    훈련, 검증, 테스트의 데이터 세트 파일 이름 획득
    '''
    if channel_name in ['train', 'validation', 'eval']:
        return [os.path.join(channel, channel_name + '.tfrecords')]
    else:
        raise ValueError('Invalid data subset "%s"' % channel_name)


def train_input_fn(hvd):
    '''
    후련의 데이터 준비
    '''
    return _input(args.epochs, args.train_batch_size, args.train, 'train', hvd)


def eval_input_fn(hvd):
    '''
    eval 데이터 준비
    '''
    return _input(args.epochs, args.eval_batch_size, args.eval, 'eval', hvd)


def validation_input_fn(hvd):
    '''
    검증 데이터 준비
    '''
    return _input(args.epochs, args.validation_batch_size, args.validation, 'validation', hvd)


def _input(epochs, batch_size, channel, channel_name, hvd):
    '''
    TFRecord 에서 데이터를 읽어서 파싱하고, 최종 데이터 세트를 제공함.
    '''
    

    filenames = get_filenames(channel_name, channel)
    dataset = tf.data.TFRecordDataset(filenames)

    # TFRecord 갯수 세기
    ds_size = sum(1 for _ in dataset)  
    if hvd.local_rank() == 0:    
        print("\n################# Loading Dataset ################")            
        print(f"\nChannel Name: {channel_name}\n")     
        print("# of batches loading TFRecord : {0}".format(ds_size)) 
    
    # 데이터 세트 파싱
    dataset = dataset.map(_dataset_parser, num_parallel_calls=10)
    dataset = dataset.repeat(1)    # 1번 반복
    
    # 후련 데이터의 경우에는 셔블링을 전체 레코드 사이즈로 함.
    if channel_name == 'train':
        buffer_size = ds_size # 셔블리의 사이즈를 전체 레코드 갯수로 제공 함.
        dataset = dataset.shuffle(buffer_size=buffer_size)
 
        if hvd.local_rank() == 0:    
            print("buffer_size: ", buffer_size)
    
    # 데이터 세트를 배치 사이즈로 나누고, 캐싱
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE) # 캐싱 함

    return dataset



def _train_preprocess_fn(image):
    """Preprocess a single training image of layout [height, width, depth]."""
    # Resize the image to add four extra pixels on each side.
    image = tf.image.resize_with_crop_or_pad(image, HEIGHT + 8, WIDTH + 8)

    # Randomly crop a [HEIGHT, WIDTH] section of the image.
    image = tf.image.random_crop(image, [HEIGHT, WIDTH, DEPTH])

    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)

    return image


def _dataset_parser(value):
    """Parse a CIFAR-10 record from value."""
    featdef = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }

    example = tf.io.parse_single_example(value, featdef)
    image = tf.io.decode_raw(example['image'], tf.uint8)
    image.set_shape([DEPTH * HEIGHT * WIDTH])

    # Reshape from [depth * height * width] to [depth, height, width].
    image = tf.cast(
        tf.transpose(tf.reshape(image, [DEPTH, HEIGHT, WIDTH]), [1, 2, 0]),
        tf.float32)
    label = tf.cast(example['label'], tf.int32)
    image = _train_preprocess_fn(image)
#    return image, tf.one_hot(label, NUM_CLASSES)
    return image, label    # 원핫 인코딩 하지 않고 실레 레이블 값을 넘김

def save_model(model, output):
    '''
    모델 저장
    '''
    tf.saved_model.save(model, output+'/1/')
    logging.info("Model successfully saved at: {}".format(output))
    return



                 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    #############################
    # 세이지 메이커 관련 환경 변수
    #############################            
    
    parser.add_argument(
        '--train',
        type=str,
        required=False,
        default=os.environ.get("SM_CHANNEL_TRAIN"),        
        help='The directory where the CIFAR-10 input data is stored.')
    parser.add_argument(
        '--validation',
        type=str,
        required=False,
        default=os.environ.get("SM_CHANNEL_VALIDATION"),        
        help='The directory where the CIFAR-10 input data is stored.')
    parser.add_argument(
        '--eval',
        type=str,
        required=False,
        default=os.environ.get("SM_CHANNEL_EVAL"),        
        help='The directory where the CIFAR-10 input data is stored.')
    parser.add_argument(
        '--model_dir',
        type=str,
        required=True,
        help='The directory where the model will be stored.')
    parser.add_argument("--model_output_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))    
    
    #############################
    # 모델 하이퍼 파라미터 
    #############################            
    
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=2e-4,
        help='Weight decay for convolutions.')
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help="""\
        This is the inital learning rate value. The learning rate will decrease
        during training. For more details check the model_fn implementation in
        this file.\
        """)
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='The number of steps to use for training.')
    parser.add_argument(
        '--train-batch-size',
        type=int,
        default=128,
        help='Batch size for training.')
    parser.add_argument(
        '--validation-batch-size',
        type=int,
        default=128,
        help='Batch size for training.')    
    parser.add_argument(
        '--eval-batch-size',
        type=int,
        default=128,
        help='Batch size for training.')        
    parser.add_argument(
        '--optimizer',
        type=str,
        default='adam')
    parser.add_argument(
        '--momentum',
        type=float,
        default='0.9')
    parser.add_argument(
        '--print-interval',
        type=int,
        default=10)

    
    args = parser.parse_args()
    main(args)