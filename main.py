import cv2
import os
from PIL import Image
import numpy as np


def get_image_data():
    sub_dir_paths = [os.path.join('./data', f) for f in os.listdir("./data")]
    faces = []
    ids = []
    for sub_dir in sub_dir_paths:
        sub_dir_list = os.listdir(sub_dir)
        for sub_dir_item in sub_dir_list:
            i_id = int(sub_dir.split('_')[1])
            ids.append(i_id)
            path = os.path.join(sub_dir, sub_dir_item)
            large_image = Image.open(path).convert('L')
            image_np = np.array(large_image, 'uint8')
            faces.append(image_np)
    return np.array(ids), faces


def create_sample_directory():
    sample_count = len(os.listdir("data"))
    sample_directory = "./data/Subject_{}".format(sample_count)
    print(sample_directory)
    os.mkdir(sample_directory)
    return sample_directory


def generate_classifier():
    id_s, face_s = get_image_data()
    lbph_classifier = cv2.face.LBPHFaceRecognizer_create()
    lbph_classifier.train(face_s, id_s)
    lbph_classifier.write('lbh_classifier.yml')


cam_port = 0
camera = cv2.VideoCapture(cam_port)
font = cv2.FONT_HERSHEY_COMPLEX
point = [200, 100, 430, 390]
img_count = 0
subject_dir = ""
samples_list = "samples.txt"

print('Press 1 - To train new face')
print('Press 2 - To test face')
choice = input("> ")

if choice == '1':
    sample_name = input("Sample Name: ")

    file_open_w = open(samples_list, "a")
    file_open_w.write("\n")
    file_open_w.write(sample_name)
    file_open_w.close()

    isExists = os.path.exists("./data")
    if not isExists:
        os.mkdir("data")
        subject_dir = create_sample_directory()
    else:
        subject_dir = create_sample_directory()

    while True:
        result, frame = camera.read()

        cv2.rectangle(frame, (430, 390), (200, 100), (0, 255, 0), 2)
        cv2.putText(frame, "Keep your face inside the rectangle then press 'c' to capture image.", (10, 30),
                    font, 0.5, (0, 255, 255), 1)

        cv2.imshow('Training', frame)

        pressedKey = cv2.waitKey(1) & 0xFF

        if pressedKey == ord('c'):
            roi = frame[point[1]:point[3], point[0]:point[2]]
            image = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            img_name = "Face_{}.jpg".format(img_count)
            cv2.imwrite("{}/{}".format(subject_dir, img_name), image)
            print("{} written!".format(img_name))
            img_count += 1

        if pressedKey == ord('q'):
            break

    generate_classifier()

if choice == '2':
    face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read("lbh_classifier.yml")
    height, width = 290, 230

    file_open_r = open(samples_list, "r+")
    samples_name_list = file_open_r.read().split("\n")
    file_open_r.close()

    while True:
        connected, image = camera.read()
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detections = face_detector.detectMultiScale(img_gray, scaleFactor=1.4, minSize=(40, 40))

        for (x, y, w, h) in detections:
            image_face = cv2.resize(img_gray[y:y + w, x:x + h], (width, height))
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            iid, confidence = face_recognizer.predict(image_face)
            name = samples_name_list[iid + 1]
            cv2.putText(image, name, (x, y + (w + 30)), font, 0.9, (0, 0, 255))
            cv2.putText(image, str(confidence), (x, y + (h + 50)), font, 0.5, (0, 0, 255))

        cv2.imshow("Face", image)
        if cv2.waitKey(1) == ord('q'):
            break

camera.release()
cv2.destroyAllWindows()
