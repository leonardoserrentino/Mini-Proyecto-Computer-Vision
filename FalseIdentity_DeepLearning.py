#Import delle librerie necessarie
import mediapipe as mp
import cv2
import pandas as pd
import numpy as np
import math

#Queste sono i landmarks costanti che vanno a delimitare una faccia
sx_face = 127
dx_face = 356
up_face = 10
down_face = 152

name = 'person'
ext = 'jpg'

#----------------------------------------------------------------------------------------

#Definizione di metodi utili per dopo

'''Questo metodo, date le immagini con cui fare il merging,
riscala, riposizione e ruota l'immagine segmentata seguendo i requisiti 
dell'immagine di background.'''
def overlay_object(bg_image, fg_image, position=(0, 0), scale=1.0, angle=0, desired_height=None):
    fg_height, fg_width = fg_image.shape[:2]
    fg_resized = cv2.resize(fg_image, (int(fg_width * scale), int(fg_height * scale)))

    if angle != 0:
        center = (fg_resized.shape[1] // 2, 0)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        fg_resized = cv2.warpAffine(fg_resized, rotation_matrix, (fg_resized.shape[1], fg_resized.shape[0]))

    fg_height, fg_width = fg_resized.shape[:2]

    if desired_height is not None and fg_height > 0:
        # Calcola il rapporto per lo stretch lungo l'asse y
        stretch_ratio = desired_height / fg_height #*1.3
        if stretch_ratio > 0:
            fg_resized = cv2.resize(fg_resized, (int(fg_width * stretch_ratio), desired_height))

    fg_height, fg_width = fg_resized.shape[:2]

    y_pos, x_pos = position[0], position[1] - fg_width // 2

    y_end, x_end = min(y_pos + fg_height, bg_image.shape[0]), min(x_pos + fg_width, bg_image.shape[1])

    # Adatta le dimensioni dell'immagine di sfondo e dell'oggetto da sovrapporre
    bg_image_roi = bg_image[max(y_pos, 0):y_end, max(x_pos, 0):x_end]
    fg_image_roi = fg_resized[max(-y_pos, 0):bg_image_roi.shape[0] + max(-y_pos, 0), max(-x_pos, 0):bg_image_roi.shape[1] + max(-x_pos, 0)]

    # Crea una maschera basata sull'oggetto non nero
    mask = cv2.cvtColor(fg_image_roi, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    # Sovrappone l'oggetto sull'immagine di sfondo usando la maschera
    for c in range(0, 3):
        bg_image_roi[:, :, c] = np.where(mask == 255, fg_image_roi[:, :, c], bg_image_roi[:, :, c])

    return bg_image


def getAngle(point1, point2):
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    angle = np.arctan2(dy, dx) * 180 / np.pi  # Converte in gradi

    return angle


def getDistanza(punto1, punto2):
    x1, y1 = punto1
    x2, y2 = punto2
    distanza = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distanza


def scalaDistanza(distanza1, distanza2):
    return distanza1/distanza2


def aggiungi_spazio_nero(img, width):    
    # Ottieni le dimensioni dell'immagine
    altezza, _, _ = img.shape
    
    # Crea uno spazio nero della dimensione desiderata
    spazio_nero = np.zeros((altezza, width, 3), dtype=np.uint8)
    
    # Concatena l'immagine originale con lo spazio nero a sinistra e destra
    nuova_immagine = np.concatenate((spazio_nero, img, spazio_nero), axis=1)

    return nuova_immagine


#----------------------------------------------------------------------------------------



'''In questa parte di codice viene inserita un'immagine
e ne viene segmentata la faccia presente.
Vengono inoltre segnate le distanze tra i landmark che ci serviranno
per posizionare e scalare l'immagine segmentata durante la 
sovrapposizione del video'''


#Carica la immagine
image = cv2.imread(f"images/{name}.{ext}")

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
    cv2.imwrite('images/partial.png', face)

segmented_f = cv2.imread('images/partial.png')
agg = aggiungi_spazio_nero(segmented_f, 500)
cv2.imwrite('images/segmented_face.png', agg)


#-------------------------------------------------------------------------------

img = cv2.imread("images/segmented_face.png")

height, width, _ = img.shape  # Ottieni le dimensioni dell'immagine

#Modello di Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

#Ricavo i lanmarks della immagine
results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
landmarks = results.multi_face_landmarks[0]

sx_face_x, sx_face_y = int(landmarks.landmark[sx_face].x * width), int(landmarks.landmark[sx_face].y * height)
dx_face_x, dx_face_y = int(landmarks.landmark[dx_face].x * width), int(landmarks.landmark[dx_face].y * height)
up_face_x, up_face_y = int(landmarks.landmark[up_face].x * width), int(landmarks.landmark[up_face].y * height)

segmented_sx = (sx_face_x, sx_face_y)
segmented_dx = (dx_face_x, dx_face_y)
segmented_up = (up_face_x, up_face_y)

fixed_distance_segmented = getDistanza(segmented_sx, segmented_dx)

#Scarico i landmarks predefiniti di Facemesh per l'ovale della faccia
face_oval = mp_face_mesh.FACEMESH_FACE_OVAL
 
#Creo un dataframe con i landmarks che mi servono, per fare operazioni piu veloci
df = pd.DataFrame(list(face_oval), columns = ["p1", "p2"])

#Lista dove salvare le connessione tra gli indici dei landmarks
routes_idx = []
 
p1 = df.iloc[0]["p1"]
p2 = df.iloc[0]["p2"]

#Allineo il datafram in modo che le coppie siano adiacenti e condividano un landmark
for i in range(0, df.shape[0]):     
    obj = df[df["p1"] == p2]
    p1 = obj["p1"].values[0]
    p2 = obj["p2"].values[0]
     
    route_idx = []
    route_idx.append(p1)
    route_idx.append(p2)
    routes_idx.append(route_idx)

#Lista per contenere le connessioni finali e normalizzate
routes = []
 
for source_idx, target_idx in routes_idx:
    #Ricavo la posizione di ogni landmark
    source = landmarks.landmark[source_idx]
    target = landmarks.landmark[target_idx]

    #Adatto la posizione assoluta del landmark (0,1) € R, alle misure dell'immagine
    relative_source = (int(img.shape[1] * source.x), int(img.shape[0] * source.y))
    relative_target = (int(img.shape[1] * target.x), int(img.shape[0] * target.y))
 
    #cv2.line(img, relative_source, relative_target, (255, 255, 255), thickness = 2)
     
    routes.append(relative_source)
    routes.append(relative_target)

#Creo la maschera con i landmark relativi estratti
mask = np.zeros((img.shape[0], img.shape[1]))
mask = cv2.fillConvexPoly(mask, np.array(routes), 1)
mask = mask.astype(bool)
  
#Creo un'immagine nera e pongo l'AND logico della maschera con la immagine sorgente
out = np.zeros_like(img)
out[mask] = img[mask]


cv2.imwrite('images/segmented.png', out)


#----------------------------------------------------------------------------------------



segmented = cv2.imread("images/segmented.png")

#Parte la capture della webcam
cap = cv2.VideoCapture(0)

ret = True
while(ret):
    ret, background = cap.read()

    height, width, _ = background.shape  # Ottieni le dimensioni dell'immagine

    # Esegui la rilevazione del volto
    results = face_mesh.process(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))

    # Estrai le coordinate degli occhi, del naso e della bocca
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            
            #Face Landmarks Coordinates
            sx_face_x, sx_face_y = int(landmarks[sx_face].x * width), int(landmarks[sx_face].y * height)
            dx_face_x, dx_face_y = int(landmarks[dx_face].x * width), int(landmarks[dx_face].y * height)
            up_face_x, up_face_y = int(landmarks[up_face].x * width), int(landmarks[up_face].y * height)
            down_face_x, down_face_y = int(landmarks[down_face].x * width), int(landmarks[down_face].y * height)
        
        pt_sx = (sx_face_x,sx_face_y)
        pt_dx = (dx_face_x,dx_face_y)
        pt_up = (up_face_x, up_face_y)
        angle = getAngle(pt_sx, pt_dx)
        scale = scalaDistanza(getDistanza(pt_sx,pt_dx), fixed_distance_segmented)
        desired_height = up_face_y-down_face_y

        position = (pt_up[1], pt_up[0])
        background = overlay_object(background, segmented, position, scale, -angle, desired_height)


    # Display the output 
    cv2.imshow("Stream", background)
    
    if cv2.waitKey(1) == 27:  # Premi Esc per uscire
        break    

cap.release()
cv2.destroyAllWindows()