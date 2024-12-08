import cv2
import numpy as np
import time
import tflite_runtime
from tflite_runtime.interpreter import Interpreter
import RPi.GPIO as GPIO
import time
object_present=False
GPIO.setmode(GPIO.BCM)
TRIG=17
ECHO=27
GPIO.setup(TRIG,GPIO.OUT)
GPIO.setup(ECHO,GPIO.IN)
interpreter=Interpreter(model_path="model-2.tflite")
interpreter.allocate_tensors()
input_details=interpreter.get_input_details()
output_details=interpreter.get_output_details()
print(input_details)
print(output_details)

def preprocess_image(image):
    input_shape=input_details[0]['shape']
    height, width, _ = image.shape
   # Define the middle crop region
    crop_height = int(height * 0.5)  # Crop to 50% height
    crop_width = int(width * 0.5)    # Crop to 50% width
    start_y = (height - crop_height) // 2
    start_x = (width - crop_width) // 2
    cropped_image = image[start_y:start_y + crop_height, start_x:start_x + crop_width]
    resized_image=cv2.resize(image,(input_shape[1],input_shape[2]))
    input_data=np.expand_dims(resized_image,axis=0).astype('float32')/255.0
    return input_data


def classify_image(image):
    input_data=preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()
    output_data=interpreter.get_tensor(output_details[0]['index'])
    predicted_label=np.argmax(output_data)
    confidence=np.max(output_data)
    return predicted_label,confidence

try:
    while True:
        GPIO.output(TRIG,False)
        time.sleep(1)
        GPIO.output(TRIG,True)
        time.sleep(0.00001)
        GPIO.output(TRIG,False)
        
        while GPIO.input(ECHO)==0:
            pulse_start=time.time()
        
        while GPIO.input(ECHO)==1:
            pulse_end=time.time()
        
        pulse_duration=pulse_end-pulse_start
        distance=pulse_duration*17150
        if(distance<10):
            print(f"Object Detected at {distance:.2f} cm")
            object_present=True
            break

except KeyboardInterrupt:
    print("Exiting")    
        
if object_present:
    camera=cv2.VideoCapture(0)
    if not camera.isOpened():
        print("cannot open the camera")
        exit()
    print("Allowing the camera to adjust to lighting conditions...")
    time.sleep(2)  # 2-second delay for brightness adjustment
    ret,frame=camera.read()   
    if not ret:
        print("error in the frame")
        camera.release()
        exit()
    cv2.imshow("live feed",frame)

    output_frame=frame.copy()
    path="/home/pi/capturedimage.jpg" # change path to location to inside folder
    cv2.imwrite(path,frame)
    print("imagge captured and send to the model")
    image=cv2.imread(path)
    predicted,confidence=classify_image(image)
    if predicted==0:
        print(f"Detected to be Biodegradable with confidence of {confidence*100} ")
        s="BIODEG"
        cv2.putText(image, f"Label: {s}, Confidence: {confidence*100:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)
    elif predicted==1:
        s="NONBIO"
        print(f"Detected to be Non Biodegradable")
        cv2.putText(image, f"Label: {s}, Confidence: {confidence*100:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)
    cv2.imwrite(image,path)
