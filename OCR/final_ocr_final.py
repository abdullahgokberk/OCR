import requests
from io import BytesIO
import cv2
import numpy as np
import pytesseract
from PIL import Image
from flask import Flask, jsonify, request
from pdf2image import convert_from_bytes


def pdf_to_text(pdf_url):
    response = requests.get(pdf_url)
    pages = convert_from_bytes(response.content)

    texts = []
    for i, page in enumerate(pages):
        img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        im_bw = cv2.threshold(gray,230,255,cv2.THRESH_BINARY)[1]
        text = pytesseract.image_to_string(Image.fromarray(im_bw), lang="tur")
        texts.append(text)

    return texts

def jpg_to_text(jpg_url):
    response = requests.get(jpg_url)
    img = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    im_bw = cv2.threshold(gray,170,255,cv2.THRESH_BINARY)[1]
    text = pytesseract.image_to_string(Image.fromarray(im_bw), lang="tur")
    return text

app = Flask(__name__)

@app.route('/ocr', methods=['POST'])
def ocr():
    request_data = request.get_json()
    doc_type = request_data['doc_type']
    image_url = request_data['image_url']

    if doc_type == "pdf":
        texts = pdf_to_text(image_url)
    elif doc_type == "jpg":
        texts = [jpg_to_text(image_url)]

    return jsonify({'texts': texts})

if __name__ == '__main__':
    app.run(port=9090)