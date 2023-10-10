import streamlit as st
from Model import YoloModel
import cv2
import numpy as np
from PIL import Image
import tempfile

weights_path = './files/yolov3.weights'
configuration_path = './files/yolov3.cfg'
labels_path = './files/coco.names'
img_path = "./images/city_scene.jpg"

with open(labels_path, 'r') as file:
    labels = file.read().strip().split('\n')

yolo = YoloModel(weights_path, configuration_path, labels_path, prob_min=0.6, threshold=0.3)

st.set_page_config(page_title="Projeto YOLO - MS960")

with st.sidebar:  
    img = st.file_uploader("Escolha uma imagem:")

    st.write("Configurações (Opcional)")

    prob_min = st.number_input("Probabilidade Minima:", min_value=0.1, max_value=1.0, value=0.6)
    threshold = st.number_input("Threshold:", min_value=0.1, max_value=0.9, value=0.3)

    btn = st.button("Detectar Objetos")

st.title("Projeto YOLO - Demonstração.")

st.divider()

if btn:
    try:
        # Salve a imagem carregada em um arquivo temporário
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_img:
            temp_img.write(img.read())
            temp_img_path = temp_img.name

        # Carregue a imagem usando o caminho temporário
        image_data = np.array(Image.open(temp_img_path))

        st.header("Imagem Original")
        st.image(image_data)

        imagem, numbers, indices = yolo.processImage(temp_img_path, probability_minimum=prob_min, threshold=threshold)

        st.header("Imagem Após Detecção")
        st.divider()
        st.image(imagem, channels="BGR")

        st.write(f"Foram encontrados {numbers} Objetos.")

        index_list = []
        for indice in indices:
            if indice not in index_list:
                st.write(f"Foram encontrados {indices.count(indice)} de {labels[indice]}")
                index_list.append(indice)
    except:
        st.warning("Erro. Verifique a imagem.")