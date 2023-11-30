import os
import cv2
import numpy as np
import onnxruntime
from time import time
from YOLOseg import YOLOseg
#from io import BytesIO
from PIL import Image
import streamlit as st


# ------------------------------------------------------------

model_q_path_40ep = 'best_q.onnx'
model_q_path_80ep = 'best_q_v2.onnx'
model_path_40ep = 'best_40ep.onnx'
model_path_80ep = 'best_80ep.onnx'
#conf_thres=0.5
#iou_thres=0.3

# ------------------------------------------------------------


# Streamlit Components
st.set_page_config(
    page_title='Document Scanner',
    page_icon=':smile:', 
    layout='wide',  # centered, wide
    # initial_sidebar_state="expanded",
    menu_items={'About': '### SantonioTheFirst',},
)


@st.cache
def load_model(model_path):
    model = YOLOseg(model_path)
    return model


def process_output_masks(image, masks):
    result = []
    masks = masks.astype(np.uint8)
    for i, mask in enumerate(masks):
        #contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
        #black = np.zeros_like(mask)
        #cv2.drawContours(black, (contour, ), -1, (1), -1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel=kernel, iterations=5)
        st.image(np.stack((opening * 255,) * 3, axis=-1), caption='smoothed by morphology opening mask')
        cropped = (np.stack((opening,) * 3, axis=-1) * image)
        st.image(cropped, caption='cropped by smoothed mask image', channels='BGR')
        rectangle = np.zeros_like(mask)
        (x, y, w, h) = cv2.boundingRect(mask)
        if w > 80 and h > 80:
            cv2.rectangle(rectangle, (x, y), (x + w, y + h), (1), -1)
            median_values = np.median(cropped[y : y + h, x : x + w, :], axis=[0, 1]).astype(np.uint8).tolist()
        area_to_fill = np.stack((np.abs(rectangle - opening),) * 3, axis=-1)
        st.image(area_to_fill * 255, caption='area to fill with median value')
        filled = (area_to_fill * median_values).astype(np.uint8)
        st.image(filled, caption='smoothed mask and bounding rectangle difference filled with median value', channels='BGR')
        restored_corners = filled + cropped
        st.image(restored_corners, caption='restored corners by filling with median value', channels='BGR')
        document = (restored_corners[y : y + h, x : x + w, :]).astype(np.uint8)
        st.image(document, 'document cropped by x, y of bounding rectangle', channels='BGR')
        document = cv2.copyMakeBorder(document, *[50 for _ in range(4)], cv2.BORDER_CONSTANT, value=median_values) #, value=[0, 0,])
        st.image(document, 'document with border filled with median value', channels='BGR')
        #document = cv2.cvtColor(document, cv2.COLOR_BGR2RGB)
        result.append(document)
    return result


#model = YOLOseg(model_path, conf_thres=conf_thres, iou_thres=iou_thres)


def main(input_file, model, conf_thres, iou_thres):
    file_bytes = np.asarray(bytearray(input_file.read()), dtype=np.uint8)  # Read bytes
    image = cv2.imdecode(file_bytes, 1)
    col1, col2 = st.columns((1, 1))
    with col1:
        st.title('Input')
        st.image(image, channels='BGR', use_column_width=True)
    with col2:
        st.title('Scanned')
        start = time()
        boxes, scores, class_ids, masks = model(image, conf_thres, iou_thres)
        # Draw detections
        combined_img = model.draw_masks(image)
        st.info(f'Prediction time: {time() - start}s')
        st.image(combined_img, channels='BGR', use_column_width=True)
        cropped_images = process_output_masks(image, masks)
        for im in cropped_images:
            st.image(im, channels='BGR', use_column_width=True)
        st.info(f'Total time: {time() - start}s')

        #if combined_img is not None:
        #    result = Image.fromarray(combined_img.astype('uint8'), 'RGB')
            #img = Image.open(result)
        #    buf = BytesIO()
        #    result.save(buf, format='PNG')
        #    byte_img = buf.getvalue() 
        #    st.download_button(
        #        label='Download image',
        #        data=byte_img,
        #        mime='image/png'
        #    )

'''
# Document scanner
'''
file_upload = st.file_uploader('Upload Document Image:', type=['jpg', 'jpeg', 'png'])

if file_upload is not None:
    #if st.checkbox('Use v2 model weights (40 + 40) epochs'):
    #    model = YOLOseg(model_path_v2) #load_model(model_path)
    #else:
    #    model = YOLOseg(model_path_v1) #load_model(model_path)
    model_list = ['Nano, 40 epochs', 'Nano, 80 epochs', 'Nano, 40 epochs, quantized', 'Nano, 80 epochs, quantized']
    option = st.selectbox(
        'What model would you like to use?',
        model_list
    )
    if option == model_list[0]:
        model_path = model_path_40ep
    elif option == model_list[1]:
        model_path = model_path_80ep
    elif option == model_list[2]:
        model_path = model_q_path_40ep
    elif option == model_list[3]:
        model_path = model_q_path_80ep
    else:
        model_path = model_q_path_80ep
    model = YOLOseg(model_path) 
    if st.checkbox('Confidence $\in$ [0.0, 100.0]'):
        conf_max_val = 100.0
        conf_step = 0.1
    else:
        conf_max_val = 1.0
        conf_step = 0.01
    conf_thres = st.slider('Confidence threshold', min_value=0.0, max_value=conf_max_val, value=0.3, step=conf_step)
    iou_thres = st.slider('Intersection over union threshold for non maximum suppresion', min_value=0.0, max_value=1.0, value=0.3, step=0.01)
    info = f'''Confidence threshold: {conf_thres},
    IoU: {iou_thres}'''
    st.info(info)
    #if st.button('Predict with params', type='primary'):
    _ = main(file_upload, model, conf_thres, iou_thres)
