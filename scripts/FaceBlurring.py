#Importing libraries
import cv2

#Starting video streaming
cap = cv2.VideoCapture(0)

#Just to start the cicle
x=100
y=100
w=100
h=100


ret = True
while(ret):
    #Getting the frame to analyze
    ret, image = cap.read()
    
    #Importing Face Detector whit haarcascade pre-trained weights
    face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
    face_data = face_detect.detectMultiScale(image, 1.3, 5)

    #Block for static blurring
    if y>50:
        y=y-50
        h=h+100
    #Defining the dimension of the face to blur
    roi = image[y:y+h, x:x+w] 
    #Applying gaussian blurring to the roi portion
    roi = cv2.GaussianBlur(roi, (75, 75), 150) 
    #Impose this blurred image on original image
    image[y:y+roi.shape[0], x:x+roi.shape[1]] = roi

    for (x, y, w, h) in face_data:
        if y>50:
            y=y-50
            h=h+100
        #Defining the dimension of the face to blur
        roi = image[y:y+h, x:x+w] 
        #Applying gaussian blurring to the roi portion
        roi = cv2.GaussianBlur(roi, (75, 75), 150) 
        #Impose this blurred image on original image
        image[y:y+roi.shape[0], x:x+roi.shape[1]] = roi 

    # Display the output 
    cv2.imshow("Stream", image)
    
    if cv2.waitKey(1) == 27:  # Premi Esc per uscire
        break


cap.release()
cv2.destroyAllWindows()
