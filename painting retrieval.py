import cv2 as cv
import numpy as np
import csv
import json
import os
import argparse
import sys

# USAGE: python retrieval -i image.png

# Funzione che crea un json dato un CSV
def csvtojson(csvfile, jsonfile):
    csvfile = open(csvfile, 'r')
    jsonfile = open(jsonfile, 'w')
    fieldnames = ("Title","Author","Room","Image")
    reader = csv.DictReader(csvfile, fieldnames)
    final = {}
    out = [ row for row in reader][1:] # Non considero la prima riga in cui non ci sono informazioni interessanti
    for c, item  in enumerate(out):
        if c < 10:
            num = "00" + str(c)
        else:
            num = "0" + str(c)
        final[num] = item 
    final = json.dumps(final)
    jsonfile.write(final)


# Funzione che dato il path di un'immagine crea una ranked list delle immagini più simili nel DB
def recognize_painting(image):
    if os.path.isfile('datasets/data_orb.json'):
        with open('datasets/data_orb.json', 'r') as fin:
            data_orb = json.load(fin)
    else:
        print("File data_orb.json non trovato")
        sys.exit()
    #query_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY) # Se prendo in input direttamente l'immagine
    query_img = cv.imread(image, cv.IMREAD_GRAYSCALE) # Se devo leggere l'immagine

    # resize image if it is too big
    if query_img.shape[0] * query_img.shape[1] > 9e5:
        query_img = cv.resize(query_img, None, fx=0.7, fy=0.7, interpolation=cv.INTER_AREA)

    # Creo l'ORB detector
    orb = cv.ORB_create()
    # Cerco i descrittori dell'immagine di query tramite ORB
    _, des = orb.detectAndCompute(query_img, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)

    results = {}

    for key, value in data_orb.items():
        des1 = np.array(value, dtype='float32')
        matches = flann.knnMatch(np.asarray(des1,np.float32),np.asarray(des,np.float32), k=2)
        # Creo un "vettore di similarità" tramite la Lowe's ratio
        good = []
        for (m, n) in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        results[key] = len(good)

    sorted_results = sorted(results.items(), key=lambda kv: kv[1], reverse=True)
    print(sorted_results)

    with open('datasets/data.json', 'r') as fin:
        data = json.load(fin)
        print("RANKED SIMILARITY LIST")
        for i, _ in enumerate(results):
            titolo = data[sorted_results[i][0]].get('Title', None)
            autore = data[sorted_results[i][0]].get('Author', None)
            room = data[sorted_results[i][0]].get('Room', None)
            nomeim = data[sorted_results[i][0]].get('Image', None)
            print(f"***** {i+1} PLACE ***** \nNome quadro: {titolo} \nAutore: {autore} \nStanza: {room} \nNome immagine: {nomeim}\n")



# Funzione che dato il path di un'immagine restituisce il relativo descrittore (usata in fase di sviluppo)
def descriptor(imagename):
    image = cv.imread(imagename)
    query_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    orb = cv.ORB_create()
    # Cero i descrittori dell'immagione tramite ORB
    _, des = orb.detectAndCompute(query_img, None)
    print(des)
    return des




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True,
	help="path to input image")
    args = vars(ap.parse_args())
    if not os.path.isfile('datasets/data.json'):
        print('Creating new json from CSV data...')
        csvtojson('datasets/data.csv','datasets/data.json')
        print('json created successfully!')
    recognize_painting(args["input"])



if __name__ == "__main__":
    main()