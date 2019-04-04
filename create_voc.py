import argparse
import numpy as np
import cv2
import data

def create_voc(images):
    det=cv2.ORB.create()
    bow_kmeans_trainer=cv2.BOWKMeansTrainer(64, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)

    print("Detecting and computing descriptors...")
    for img in images:
        bow_kmeans_trainer.add(np.float32(det.detectAndCompute(img, None)[1]))
    voc=bow_kmeans_trainer.cluster()

    print("Done")
    print("Saving BoF...")
    fs=cv2.FileStorage("Vocabulare.xml", cv2.FILE_STORAGE_WRITE)
    fs.write("voc", voc)
    fs.release()
    print("Done")

if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=str, required=True,  help="path to img directory")
    args=vars(ap.parse_args())
    create_voc(data.get_data(args["input"]))