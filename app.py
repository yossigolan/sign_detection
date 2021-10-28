import numpy as np
from tensorflow.keras.models import load_model
import gradio as gr
# import requests

classes = {1: 'Speed limit (20km/h)',
           2: 'Speed limit (30km/h)',
           3: 'Speed limit (50km/h)',
           4: 'Speed limit (60km/h)',
           5: 'Speed limit (70km/h)',
           6: 'Speed limit (80km/h)',
           7: 'End of speed limit (80km/h)',
           8: 'Speed limit (100km/h)',
           9: 'Speed limit (120km/h)',
           10: 'No passing',
           11: 'No passing veh over 3.5 tons',
           12: 'Right-of-way at intersection',
           13: 'Priority road',
           14: 'Yield',
           15: 'Stop',
           16: 'No vehicles',
           17: 'Veh > 3.5 tons prohibited',
           18: 'No entry',
           19: 'General caution',
           20: 'Dangerous curve left',
           21: 'Dangerous curve right',
           22: 'Double curve',
           23: 'Bumpy road',
           24: 'Slippery road',
           25: 'Road narrows on the right',
           26: 'Road work',
           27: 'Traffic signals',
           28: 'Pedestrians',
           29: 'Children crossing',
           30: 'Bicycles crossing',
           31: 'Beware of ice/snow',
           32: 'Wild animals crossing',
           33: 'End speed + passing limits',
           34: 'Turn right ahead',
           35: 'Turn left ahead',
           36: 'Ahead only',
           37: 'Go straight or right',
           38: 'Go straight or left',
           39: 'Keep right',
           40: 'Keep left',
           41: 'Roundabout mandatory',
           42: 'End of no passing',
           43: 'End no passing veh > 3.5 tons'}


def classify_sign(img):
    img = img.resize((30, 30))
    img = np.expand_dims(img, axis=0)
    img = np.array(img)
    pred = model.predict_classes([img])[0]
    pred2 = model.predict_proba([img]).flatten()
    #     pred = model.predict([img]).argmax()
    return {classes[i+1]: float(pred2[i]) for i in range(len(classes)-1)}


model = load_model("my_model.h5")
image = gr.inputs.Image(type='pil', image_mode="RGB")
label = gr.outputs.Label(num_top_classes=5)
title = 'Sign Detective!'
description = 'Snap a picture of a road sign (or choose one from the samples below) and find out which sign it is!'
sample_images = [["00001.png"], ['00003.png'], ['00018.png'], ['00031.png'], ['00093.png'],
                 ['00023.png'], ['00053.png'], ['00057.png'], ['00070.png'], ['00094.png']]

gr.Interface(fn=classify_sign, inputs=image,
             outputs=label, examples=sample_images,
             capture_session=True, title=title,
             description=description).launch(debug=True, share=True)
