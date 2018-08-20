
import hypermax
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def measureAccuracy(params):
	# ### Downloasd the fashion_mnist data
	# Load the fashion-mnist pre-shuffled train data and test data
    (x_train,y_train),(x_test,y_test)=tf.keras.datasets.fashion_mnist.load_data()
	# Fashion labels
    fashion_labels=['T-shirt/Top',  #index 0
		        'Trouser',      #index 1
		        'Pullover',     #index 2
		        'Dress',        #index 3
		        'Coat',         #index 4
		        'Sandal',       #index 5
		        'Shirt',        #index 6
		        'Sneaker',      #index 7
		        'Bag',          #index 8
		        'Ankle boot'    #index 9
		       ]

	
	# ### Data normalization
    x_train=x_train.astype('float32')/255
    x_test=x_test.astype('float32')/255

	
	# ### Split the data into train/validation/test data sets
	# Split train and validation datasets: 55000 for train and 5000 for validation
    (x_train,x_valid)=x_train[5000:],x_train[:5000]
    (y_train,y_valid)=y_train[5000:],y_train[:5000]

	# reshape input data from (28,28) to (28,28,1)
    w,h=28,28
    x_train=x_train.reshape(x_train.shape[0],w,h,1)
    x_valid=x_valid.reshape(x_valid.shape[0],w,h,1)
    x_test=x_test.reshape(x_test.shape[0],w,h,1)

	# one-hot encode the labels
    y_train=tf.keras.utils.to_categorical(y_train,10)
    y_valid=tf.keras.utils.to_categorical(y_valid,10)
    y_test=tf.keras.utils.to_categorical(y_test,10)

	
	# Create the model architecture
	# 2 convolutional nerual networks
	# 2 max pooling layers
	# 2 dropout layers
	# 1 fully connected layers
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(
		                        filters=int(params['layer1_filters']),
		                        kernel_size=int(params['layer1_filter_size']),
		                        padding='same',
		                        activation=params['activation'],
		                        input_shape=(28,28,1)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(float(params['layer1_dropout'])))
    
    model.add(tf.keras.layers.Conv2D(
		                        filters=int(params['layer2_filters']),
		                        kernel_size=int(params['layer2_filter_size']),
		                        padding='same',
		                        activation=params['activation']))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(float(params['layer2_dropout'])))
    
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(int(params['fully_connected_size']), activation=params['activation']))
    model.add(tf.keras.layers.Dropout(float(params['fully_connected_dropout'])))
    model.add(tf.keras.layers.Dense(10,activation='softmax'))

	#print('Model summary:')
    model.summary()


	# ### Compile the model
    model.compile(loss='categorical_crossentropy',
		     optimizer='adam',
		     metrics=['accuracy'])


	# ### Train the model & save the model

	# Moodel check pointer
    from keras.callbacks import ModelCheckpoint
    from keras.callbacks import EarlyStopping
    
    checkpointer=ModelCheckpoint(filepath='model.weigths.best.hdf5',
		                     verbose=1,
		                    save_best_only=True)
    early_stop=EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1)

	# Train the model
    model.fit(x_train,
		 y_train,
		 batch_size=64,
		 epochs=50,
		 validation_data=(x_valid,y_valid),
		 callbacks=[checkpointer,early_stop])

	# ### Load model with the best validation accuracy
    model.load_weights('model.weigths.best.hdf5')


	# ### Test Accuracy
	# Evaluate the mode on test set
    score=model.evaluate(x_test,y_test,verbose=0)
    y_hat=model.predict(x_test)
    
    return {"loss": (1.0 - score[1])}
