import cv2
import os
import numpy as np


def preprocesing(class_vector, input_folder, output_file, x_resize, y_resize):
    dataset = []
    for dirname, dirnames, filenames in os.walk(input_folder):
        for filename in filenames:
            file = os.path.join(dirname, filename)
            img = cv2.imread(file)
            img = cv2.resize(img, (0, 0), fx=x_resize, fy=y_resize)  # img reduction
            sample = []
            for row in img:
                for pixel in row:
                    sample.append(
                        pixel[0] / 255
                    )  # we only take a value of RGB because it is grayscale and we divide by 255 to normalize
            # a cada muestra le agregamos la clase codificada con one hot encoding
            # en cada carpeta la clase debiera ser diferente
            class_vec = class_vector
            # a la muestra le agregamos la clase al final
            sample.extend(class_vec)
            # luego agregamos la muestra con su clase al dataset
            dataset.append(sample)
            # estas cuatro últimas líneas las puedes comentar cuando ejecutes el script
            # con esto la arquitectura de la red tiene 4096 de entrada y 4 de salida
            # las capas intermedias las puedes adaptar desde los ejemplos del MNIST
            print(len(sample))
            cv2.imshow("image", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    output = open(output_file, "w")
    for line in dataset:
        line = list(map(str, line))
        s = ";".join(line)
        output.write(s + "\n")
    output.close()
