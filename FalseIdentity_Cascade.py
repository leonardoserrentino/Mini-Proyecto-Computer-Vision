#Importing libraries
import cv2

#Starting video streaming
cap = cv2.VideoCapture(0)

task = int(input("Do you want to take an image or upload here one? 1 or 2 "))
if task == 1:
    #Check if camera is opened
    if not cap.isOpened():
        print("Opening camera failed")
        exit()

    #Wait for the user to take the shot
    input("Press Enter to take a shot...")

    #Get a single frame of the camera
    ret, frame = cap.read()

    #Check if the shot did well
    if not ret:
        print("Shot failed")
        cap.release()
        exit()

    #Saving the image in the directory
    cv2.imwrite('images/image.png', frame)

    #Upload the image to segment the face from
    image = cv2.imread('images/image.png')

elif task == 2:
    path = input('Write the image path to segment and use for the task: ')
    #Upload the image to segment the face from
    image = cv2.imread(path)
    if image is None:
        print('Path opening failed')
        exit()
else:
    print('Task needs to be 1 or 2')
    exit()

#Importing Face Detector whit haarcascade pre-trained weights
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#Converting img to gray (less calculation)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Esegue il rilevamento del volto
face_detect = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Crea un'immagine vuota con lo stesso spazio dei colori dell'immagine originale
segmented_image = image.copy()

# Estrae e salva la faccia rilevata
for (x, y, w, h) in face_detect:
    face = segmented_image[y:y+h, x:x+w]
    cv2.imwrite('images/segmented_face.png', face)

falseIdentity = cv2.imread('images/segmented_face.png')


while(True):
    #Getting the frame to analyze
    ret, frame = cap.read()
    
    #Use the face classifier to detect tha face in this frame
    face_data = face_cascade.detectMultiScale(frame, 1.3, 5) 

    for (x, y, w, h) in face_data:
        #Defining the dimension of the face to substitute
        roi = frame[y:y+h, x:x+w] 
        ww = roi.shape[0]
        hh = roi.shape[1]
        #Resizing the false Identity image to match roi
        resized = cv2.resize(falseIdentity, [ww,hh])
        #Substitute the false identity image in the detected area
        frame[y:y+hh, x:x+ww] = resized

    # Display the output 
    cv2.imshow("Stream", frame)
    
    if cv2.waitKey(1) == 27:  # Premi Esc per uscire
        break

cap.release()
cv2.destroyAllWindows()