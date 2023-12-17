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

model_q_path_40ep = 'best_q_40ep.onnx'
model_q_path_80ep = 'best_q_80ep.onnx'
model_q_path_160ep = 'best_q_160ep.onnx'
model_path_40ep = 'best_40ep.onnx'
model_path_80ep = 'best_80ep.onnx'
model_path_160ep = 'best_160ep.onnx'
model_path_80ep_bg = 'best_80ep_bg.onnx'
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


def get_smooth_mask(mask, kernel_size=17, iterations=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel=kernel, iterations=iterations)
    return opening


def crop(img, mask):
    return np.stack((mask,) * 3, axis=-1) * img


def get_min_rectangle(mask, box):
    rectangle = np.zeros_like(mask)
    x1, y1, x2, y2 = box.astype(int)
    cv2.rectangle(rectangle, (x1, y1), (x2, y2), (1), -1)
    return rectangle, (x1, y1, x2, y2)


def get_median_values(img):
    return np.median(img, axis=[0, 1]).astype(np.uint8).tolist()


def remove_shadows(img, kernel_size=7, blur_size=21):
    rgb_planes = cv2.split(img) 
    result_planes = [] 
    result_norm_planes = [] 
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((kernel_size, kernel_size), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, blur_size)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)
    result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)
    return result_norm
            
         
def process_output_masks(image, masks, boxes, border):
    result = []
    masks = masks.astype(np.uint8)
    for (mask, box) in zip(masks, boxes):
        #contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
        #black = np.zeros_like(mask)
        #cv2.drawContours(black, (contour, ), -1, (1), -1)
        #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        #opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel=kernel, iterations=5)
        opening = get_smooth_mask(mask)
        st.image(np.stack((opening * 255,) * 3, axis=-1), caption='smoothed by morphology opening mask')
        cropped = crop(image, opening)
        st.image(cropped, caption='cropped by smoothed mask image', channels='BGR')
        #rectangle = np.zeros_like(mask)
        #(x, y, w, h) = cv2.boundingRect(mask)
        #if w > 80 and h > 80:
        #    cv2.rectangle(rectangle, (x, y), (x + w, y + h), (1), -1)
        rectangle, (x1, y1, x2, y2) = get_min_rectangle(mask, box)
        median_values = get_median_values(cropped[y1 : y2, x1 : x2, :])
        area_to_fill = np.stack((np.abs(rectangle - opening),) * 3, axis=-1)
        st.image(area_to_fill * 255, caption='area to fill with median value')
        filled = (area_to_fill * median_values).astype(np.uint8)
        st.image(filled, caption='smoothed mask and bounding rectangle difference filled with median value', channels='BGR')
        restored_corners = filled + cropped
        st.image(restored_corners, caption='restored corners by filling with median value', channels='BGR')
        document = (restored_corners[y1 : y2, x1 : x2, :]).astype(np.uint8)
        st.image(document, 'document cropped by x, y of bounding rectangle', channels='BGR')
        document = cv2.copyMakeBorder(document, *[border for _ in range(4)], cv2.BORDER_CONSTANT, value=median_values) #, value=[0, 0,])
        st.image(document, 'document with border filled with median value', channels='BGR')
        #document = cv2.cvtColor(document, cv2.COLOR_BGR2RGB)
        document = remove_shadows(document)
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
        #st.image(remove_shadows(image), channels='BGR', caption='removed shadows')
        start = time()
        boxes, scores, class_ids, masks = model(image, conf_thres, iou_thres)
        st.info(', '.join(map(str, class_ids)))
        # Draw detections
        combined_img = model.draw_masks(image)
        st.info(f'Prediction time: {time() - start}s')
        st.image(combined_img, channels='BGR', use_column_width=True)
        cropped_images = process_output_masks(image, masks, boxes, border=50)
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
    model_list = ['Nano, 40 epochs', 'Nano, 80 epochs', 'Nano, 160 epochs', 'Nano, 40 epochs, quantized', 'Nano, 80 epochs, quantized', 'Nano, 160 epochs, quantized', 'Nano, 80 epochs 3 classes']
    option = st.selectbox(
        'What model would you like to use?',
        model_list
    )
    if option == model_list[0]:
        model_path = model_path_40ep
    elif option == model_list[1]:
        model_path = model_path_80ep
    elif option == model_list[2]:
        model_path = model_path_160ep
    elif option == model_list[3]:
        model_path = model_q_path_40ep
    elif option == model_list[4]:
        model_path = model_q_path_80ep
    elif option == model_list[5]:
        model_path = model_q_path_160ep
    elif option == model_list[6]:
        model_path = model_path_80ep_bg
    else:
        model_path = model_path_80ep_bg
    st.info(f'model: {model_path}')
    model = YOLOseg(model_path) 
    if st.checkbox('Confidence $\in$ [0.0, 100.0]'):
        conf_max_val = 100.0
        conf_step = 0.1
    else:
        conf_max_val = 1.0
        conf_step = 0.01
    conf_thres = st.slider('Confidence threshold', min_value=0.0, max_value=conf_max_val, value=0.25, step=conf_step)
    iou_thres = st.slider('Intersection over union threshold for non maximum suppresion', min_value=0.0, max_value=1.0, value=0.7, step=0.01)
    info = f'''Confidence threshold: {conf_thres},
    IoU: {iou_thres}'''
    st.info(info)
    #if st.button('Predict with params', type='primary'):
    _ = main(file_upload, model, conf_thres, iou_thres)
