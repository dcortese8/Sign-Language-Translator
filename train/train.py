import numpy as np
import pathlib
import os
import random
from FrameGenerator import FrameGenerator
import cv2
import tensorflow as tf
import tensorflow_hub as hub
from CustomMovinet import CustomModel
from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model



def train(data_path='data', num_frames=8, model_id='a0'):

    #actions = np.array(['hello', 'love', 'thank you'])
    #splits = metadata['split'].unique()

    if not os.path.exists(data_path):
        raise Exception('Data path does not exist. Run preprocess.py')

    subset_paths = {
        'train': pathlib.Path(os.path.join(data_path, 'train')),
        'val': pathlib.Path(os.path.join(data_path, 'val'))
    }

    num_classes = len(next(os.walk(subset_paths['train']))[1])

    output_signature = (
        tf.TensorSpec(shape = (None, None, None, 3), dtype = tf.float32),
        tf.TensorSpec(shape = (None,), dtype = tf.uint8)
    )

    train_ds = tf.data.Dataset.from_generator(
        FrameGenerator(subset_paths['train'], num_frames),
        output_signature = output_signature
    )

    # Create the validation set
    val_ds = tf.data.Dataset.from_generator(
        FrameGenerator(subset_paths['val'], num_frames),
        output_signature = output_signature
    )

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
    val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)

    train_ds = train_ds.batch(1)
    val_ds = val_ds.batch(1)

    #tf.keras.backend.clear_session()

    # Create backbone and model
    use_positional_encoding = model_id in {'a3', 'a4', 'a5'}
    backbone = movinet.Movinet(
        model_id=model_id,
        causal=True,
        conv_type='2plus1d',
        se_type='2plus3d',
        activation='hard_swish',
        gating_activation='hard_sigmoid',
        use_positional_encoding=use_positional_encoding,
        use_external_states=False
    )

    # # Create a movinet classifier using this backbone.
    # model = movinet_model.MovinetClassifier(
    # 	backbone,
    # 	num_classes=600,
    # 	output_states=True)

    model = movinet_model.MovinetClassifier(
        backbone=backbone,
        num_classes=num_classes,
        output_states=True
    )

    movinet_hub_url = f'https://tfhub.dev/tensorflow/movinet/{model_id}/stream/kinetics-600/classification/3'

    movinet_hub_model = hub.KerasLayer(movinet_hub_url, trainable=True)

    # Input layer for the frame sequence
    image_input = tf.keras.layers.Input(
        shape=[None, None, None, 3],
        dtype=tf.float32,
        name='image'
    )

  # Input layers for the different model states.
    init_states_fn = movinet_hub_model.resolved_object.signatures['init_states']

    state_shapes = {
        name: ([s if s > 0 else None for s in state.shape], state.dtype)
        for name, state in init_states_fn(tf.constant([0, 0, 0, 0, 3])).items()
    }

    states_input = {
        name: tf.keras.Input(shape[1:], dtype=dtype, name=name)
        for name, (shape, dtype) in state_shapes.items()
    }

    # Wrap the Movinet model in a Keras model so that it can be finetuned.

    inputs = {**states_input, 'image': image_input}

    outputs = model(inputs)

    model = CustomModel(inputs, outputs, name='movinet')

    for layer in model.layers[:-1]:
        layer.trainable = False
    model.layers[-1].trainable = True

    init_states = init_states_fn(tf.shape(tf.ones([1, num_frames, 172, 172, 3])))



    def add_states(video, label, stream_states=init_states):
        return ({**stream_states, "image": video} , label)



    train = train_ds.map(add_states)
    val = val_ds.map(add_states)

    num_epochs = 3

    train_steps = 10#len(train_dataset_df) // batch_size
    total_train_steps = train_steps * num_epochs
    test_steps = 1#(len(valid_dataset_df) // batch_size) or 1

    loss_obj = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True,
        label_smoothing=0.1
    )

    metrics = [
        tf.keras.metrics.TopKCategoricalAccuracy(
            k=1, name='top_1', dtype=tf.float32
            ),
        tf.keras.metrics.TopKCategoricalAccuracy(
            k=3, name='top_3', dtype=tf.float32
            ),
    ]

#     initial_learning_rate = 0.01
#     learning_rate = tf.keras.optimizers.schedules.CosineDecay(
#         initial_learning_rate, decay_steps=total_train_steps,
#     )

#     optimizer = tf.keras.optimizers.RMSprop(
#         learning_rate, rho=0.9, momentum=0.9, epsilon=1.0, clipnorm=1.0)

    model.compile(loss=loss_obj, optimizer=tf.keras.optimizers.Adam(), metrics=metrics)

#     checkpoint_filepath = "~/Sign-Language-Translator/notebooks/movinet_checkpoints_stream"

#     model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#         filepath=checkpoint_filepath,
#         save_weights_only=True,
#         monitor='val_top_1',
#         mode='max',
#         save_best_only=True
#     )

#     logdir = "logs/train/"# + datetime.now().strftime("%Y%m%d-%H%M%S")
#     #tensorboard --logdir=./notebooks/logs

#     callbacks = [
#         tf.keras.callbacks.TensorBoard(log_dir=logdir)#,
#         #model_checkpoint_callback
#     ]

    #train_dataset = train
    #valid_dataset = val

    print("Beginning fit....")
    
    #results = model_wrapped.fit(
    results = model.fit(
    #results = model_movinet.fit(
        train,
        validation_data=val,
        epochs=20,
        #steps_per_epoch=train_steps,
        validation_steps=test_steps,
        #callbacks=callbacks,
        validation_freq=1,
        verbose=1
    )
    
if __name__ == '__main__':
  train()