# To Capture Frame
import cv2
from matplotlib.pyplot import fill

# To process image array
import numpy as np
import tensorflow as tf

# import the tensorflow modules and load the model

'''from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img

training_data_generator = ImageDataGenerator(
	rescale=1.0/255,
	rotation_range=40,
	width_shift_range=0.3,
	height_shift_range=0.3,
	zoom_range=0.3,
	horizontal_flip=True,
	vertical_flip=True,
	fill_mode='nearest'
)

training_image_paper = './Paper-samples.zip'
training_image_rock = './Rock-samples.zip'
training_image_scissors = './Scissor-samples.zip'

training = training_data_generator.flow_from_directory(
	(training_image_paper, training_image_rock, training_image_scissors),
	target_size=(180, 180)
)'''

mymodel = tf.keras.models.load_model("keras_model.h5")
# Attaching Cam indexed as 0, with the application software
camera = cv2.VideoCapture(0)

# Infinite loop
while True:

	# Reading / Requesting a Frame from the Camera 
	status , frame = camera.read()

	# if we were sucessfully able to read the frame
	if status:

		# Flip the frame
		# Flip the frame
		frame = cv2.flip(frame , 1)

		# Resize the frame
		resized_frame = cv2.resize(frame , (224,224))

		# Expanding the dimension of the array along axis 0
		resized_frame = np.expand_dims(resized_frame , axis = 0)

		# Normalizing for easy processing
		resized_frame = resized_frame / 255

		# Getting predictions from the model
		predictions = mymodel.predict(resized_frame)

		# Converting the data in the array to percentage confidence 
		rock = int(predictions[0][0]*100)
		paper = int(predictions[0][1]*100)
		scissor = int(predictions[0][2]*100)

		# printing percentage confidence
		print(f"Rock: {rock} %, Paper: {paper} %, Scissor: {scissor} %")

		# displaying the frames captured
		cv2.imshow('feed' , frame)

		# waiting for 1ms
		code = cv2.waitKey(1)
		
		# if space key is pressed, break the loop
		if code == 32:
			break

# release the camera from the application software
camera.release()

# close the open window
cv2.destroyAllWindows()
