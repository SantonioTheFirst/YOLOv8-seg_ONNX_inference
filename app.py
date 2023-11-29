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

model_path_v1 = 'best_q.onnx'
model_path_v2 = 'best_q_v2.onnx'
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
    masks *= 255
    for i, mask in enumerate(masks):
        #cropped = (np.stack((mask, ) * 3, axis=-1) * image)
        #mask = cv2.erode(mask, kernel=(3, 3), iterations=2)
        #mask = (mask * 255.0).astype(np.uint8)
        st.info(f'{mask.max()}, {mask.shape}')
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
        black = np.zeros_like(mask)
        cv2.drawContours(black, (contour, ), -1, (1), -1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        opening = cv2.morphologyEx(black, cv2.MORPH_OPEN, kernel=kernel, iterations=5)
        cropped = (np.stack((opening,) * 3, axis=-1) * image)
        #peri = cv2.arcLength(contour, True)
        #approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        #st.image(cv2.drawContours(image, contour, -1, (255), 5))
        rectangle = np.zeros_like(mask)
        (x, y, w, h) = cv2.boundingRect(contour)
        if w > 80 and h > 80:
            cv2.rectangle(rectangle, (x, y), (x + w, y + h), (255), -1)
            median_values = np.median(cropped[y : y + h, x : x + w, :], axis=[0, 1]).astype(np.uint8).tolist()
        area_to_fill = np.stack((np.abs(rectangle - mask),) * 3, axis=-1)
        filled = (area_to_fill / 255.0) * median_values
        restored_corners = filled + cropped
        document = (restored_corners[y : y + h, x : x + w, :]).astype(np.uint8)
        document = cv2.copyMakeBorder(document, *[50 for _ in range(4)], cv2.BORDER_CONSTANT, value=median_values) #, value=[0, 0,])
        #document = cv2.cvtColor(document, cv2.COLOR_BGR2RGB)
        result.append(document)
        #cv2_imshow(document)
    return result


#model = YOLOseg(model_path, conf_thres=conf_thres, iou_thres=iou_thres)


def main(input_file, model, conf_thres, iou_thres):
    file_bytes = np.asarray(bytearray(input_file.read()), dtype=np.uint8)  # Read bytes
    image = cv2.cvtColor(cv2.imdecode(file_bytes, 1), cv2.COLOR_BGR2RGB)
    col1, col2 = st.columns((1, 1))
    with col1:
        st.title('Input')
        #st.write(image.shape)
        st.image(image, channels='RGB', use_column_width=True)
    with col2:
        st.title('Scanned')
        start = time()
        boxes, scores, class_ids, masks = model(image, conf_thres, iou_thres)
        # Draw detections
        combined_img = model.draw_masks(image)
        st.info(f'Prediction time: {time() - start}s')
        st.image(combined_img, channels='RGB', use_column_width=True)
        cropped_images = process_output_masks(image, masks)
        for im in cropped_images:
            st.image(im, channels='RGB', use_column_width=True)
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
    if st.checkbox('Use v2 model weights (40 + 40) epochs'):
        model = YOLOseg(model_path_v2) #load_model(model_path)
    else:
        model = YOLOseg(model_path_v1) #load_model(model_path) 
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
