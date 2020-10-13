import requests
import time
import json
import os
from PIL import Image, ImageEnhance
def ocr_space_file(filename, overlay=False, api_key='e61f4d4c3488957', language='eng'):
    """ OCR.space API request with local file.
        Python3.5 - not tested on 2.7
    :param filename: Your file path & name.
    :param overlay: Is OCR.space overlay required in your response.
                    Defaults to False.
    :param api_key: OCR.space API key.
                    Defaults to 'helloworld'.
    :param language: Language code to be used in OCR.
                    List of available language codes can be found on https://ocr.space/OCRAPI
                    Defaults to 'en'.
    :return: Result in JSON format.
    """

    payload = {'isOverlayRequired': overlay,
               'apikey': api_key,
               'language': language,
               }
    with open(filename, 'rb') as f:
        r = requests.post('https://api.ocr.space/parse/image',
                          files={filename: f},
                          data=payload,
                          )
    return r.content.decode()
def ocr_result(image_path):
    subscription_key = "ad143190288d40b79483aa0d5c532724"
    vision_base_url = "https://westus2.api.cognitive.microsoft.com/vision/v2.0/"
    ocr_url = vision_base_url + "recognizeText?mode=Printed"
    headers = {'Ocp-Apim-Subscription-Key': subscription_key, 'Content-Type': 'application/octet-stream'}
    params = {'language': 'eng', 'detectOrientation': 'true'}

    image = Image.open(image_path)
    enh_con = ImageEnhance.Contrast(image)
    contrast = 2.0
    image = enh_con.enhance(contrast)
    # image = image.convert('L')
    # image = image.resize((800, 800))
    image.save('OCR_temp.png')
    image_data = open('OCR_temp.png', "rb").read()
    response = requests.post(ocr_url, headers=headers, params=params, data=image_data)
    response.raise_for_status()
    op_location = response.headers['Operation-Location']
    analysis = {}
    while "recognitionResults" not in analysis.keys():
        time.sleep(3)
        binary_content = requests.get(op_location, headers=headers, params=params).content
        analysis = json.loads(binary_content.decode('ascii'))
    line_infos = [region["lines"] for region in analysis["recognitionResults"]]
    word_infos = []
    for line in line_infos:
        for word_metadata in line:
            for word_info in word_metadata["words"]:
                if 'confidence' in word_info.keys():
                    if word_info['confidence'] == 'Low':
                        continue
                if word_info['boundingBox'][0] > word_info['boundingBox'][4]:
                    continue
                word_infos.append(word_info)
    return word_infos
image_path = 'C:\\work\\evalset_fqa\\vbar\\bitmap\\'
image_names = os.listdir(image_path)
for name in ['495.jpg', '151,jpg']:
    image_file_path = os.path.join(image_path, name)
    result = ocr_result(image_file_path)
    print(result)
