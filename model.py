import tensorflow as tf
import os
from data import data_builder
def build_model(args):
    data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
    ])
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    IMG_SHAPE = args.img_shape
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
    base_model.trainable = False
    prediction_layer = tf.keras.layers.Dense(1)
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    inputs = tf.keras.Input(shape=IMG_SHAPE)
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs) 
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0, name='accuracy')])
    return model

def train(args):
    net=build_model(args)
    train_data,val_data=data_builder(args)
    print("Starting Initial training")
    history=net.fit(train_data,epochs=args.initial_epochs,validation_data=val_data)
    print("initial training over")
    for layer in net.layers[args.fine_tune_start+2:]:
        layer.trainable = True
    net.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer = tf.keras.optimizers.RMSprop(learning_rate=args.lr/10),
              metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0, name='accuracy')])
    total_epochs =  args.initial_epochs + args.fine_tune_epochs
    print("Starting fine-tuning training")
    history_fine = net.fit(train_data,epochs=total_epochs,initial_epoch=history.epoch[-1],validation_data=val_data)
    print("Fine-tuning training over")
    net.save(os.path.join(args.path_to_model,'Gbg_dtction'))
    
def train(args):
    train_data=data_builder(args)
    net=tf.keras.models.load_model(args.path_to_trained_model)
    net.evaluate(train_data)