from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten
from keras.layers import Conv2D,MaxPooling2D
import tensorflow as tf

class CNN:
	def __init__(self):
		self.model = Sequential()
		self.model.add(Conv2D(32,(3,3),padding="same",input_shape=(28,28,1)))
		self.model.add(Activation('relu'))
		self.model.add(MaxPooling2D(pool_size=(2,2)))

		self.model.add(Conv2D(64,(3,3),padding='same'))	
		self.model.add(Activation('relu'))
		self.model.add(Conv2D(128,(3,3),padding='same'))
		self.model.add(MaxPooling2D(pool_size=(2,2)))

		self.model.add(Flatten())
		self.model.add(Dense(256))
		self.model.add(Activation('relu'))

		self.model.add(Dense(10))
		self.model.add(Activation('softmax'))
		self.model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
		
	
	def summary(self):
		return self.model.summary()

	def train(self,x,y):
		self.model.fit(x,y,batch_size=64,epochs=15)
		
	def score(self,x,y):
		s = self.model.evaluate(x,y,batch_size=64)
		print(f'loss: {s[0]}')
		print(f'accuracy: {s[1]}')
	
	def save(self,name):
		self.model.save(name)

	

if __name__ == '__main__':
	(x_train,y_train),(x_test,y_test) = mnist.load_data()
	print(x_train.shape,y_train.shape)
	print(x_test.shape,y_test.shape)
	x_train = x_train.reshape(60000,28,28,1)
	x_test = x_test.reshape(10000,28,28,1)
	#one-hot encoding
	y_train = np_utils.to_categorical(y_train,10)
	y_test = np_utils.to_categorical(y_test,10)
	model = CNN()
	print(model.summary())
	model.train(x_train,y_train)
	model.score(x_test,y_test)
	model.save('cnn_model.h5')
