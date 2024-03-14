import cv2
import mediapipe as mp

# Inizializzazione della cattura video dalla webcam
cap = cv2.VideoCapture(0)

# Inizializzazione del modello di Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

while True:
    # Lettura dei frame dalla webcam
    ret, frame = cap.read()

    if not ret:
        break

    # Converti il frame in RGB (mediapipe richiede l'input in formato RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Elaborazione del frame per trovare i landmarks del volto
    results = face_mesh.process(frame_rgb)

    # Disegno dei landmarks e delle linee tra i landmark adiacenti
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, landmark in enumerate(face_landmarks.landmark):
                height, width, _ = frame.shape
                x, y = int(landmark.x * width), int(landmark.y * height)

                # Disegna i punti landmark come cerchi verdi
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                '''# Disegna linee tra i landmark adiacenti per formare i triangoli
                connections = mp_face_mesh.FACEMESH_TESSELATION
                for connection in connections:
                    point1 = connection[0]
                    point2 = connection[1]
                    pt1 = (int(face_landmarks.landmark[point1].x * width), int(face_landmarks.landmark[point1].y * height))
                    pt2 = (int(face_landmarks.landmark[point2].x * width), int(face_landmarks.landmark[point2].y * height))
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 1)
'''
    # Mostra il frame con i landmarks e le linee tra i landmark adiacenti
    cv2.imshow('Face Landmarks', frame)

    # Esci dal loop se viene premuto il tasto 'Esc'
    if cv2.waitKey(1) == 27:
        break

# Rilascia la cattura video e chiudi tutte le finestre
cap.release()
cv2.destroyAllWindows()
