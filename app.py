import os
import cv2
import numpy as np
import onnxruntime
from time import time
from YOLOseg import YOLOseg
from io import BytesIO
from PIL import Image
import streamlit as st


# ------------------------------------------------------------

model_path = 'best_q.onnx'
conf_thres=0.7
iou_thres=0.3

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
def load_model(model_path, conf_thres=0.7, iou_thres=0.3):
    return YOLOseg(model_path, conf_thres, iou_thres)


model = YOLOseg(model_path, conf_thres=conf_thres, iou_thres=iou_thres)


def main(input_file, procedure):
    file_bytes = np.asarray(bytearray(input_file.read()), dtype=np.uint8)  # Read bytes
    image = cv2.cvtColor(cv2.imdecode(file_bytes, 1), cv2.COLOR_BGR2RGB)
    col1, col2 = st.columns((1, 1))
    with col1:
        st.title('Input')
        st.write(image.shape)
        st.image(image, channels='RGB', use_column_width=True)
    with col2:
        st.title('Scanned')
        if procedure == 'Traditional':
            pass
        else:
            start = time()
            boxes, scores, class_ids, masks = model(image)
            # Draw detections
            combined_img = model.draw_masks(image)
            st.info(f'Prediction time: {time() - start}s')
            st.image(combined_img, channels='RGB', use_column_width=True)

        if combined_img is not None:
            result = Image.fromarray(combined_img.astype('uint8'), 'RGB')
            img = Image.open(result)
            buf = BytesIO()
            img.save(buf, format='PNG')
            byte_img = buf.getvalue() 
            st.download_button(
                label='Download image',
                data=byte_img,
                mime='image/png'
            )

'''
# Document scanner
'''

procedure_selected = st.radio('Select Scanning Procedure:', ('Traditional', 'Deep Learning'), index=1, horizontal=True)

tab1, tab2 = st.tabs(['Upload a Document', 'Capture Document'])

with tab1:
    file_upload = st.file_uploader('Upload Document Image:', type=['jpg', 'jpeg', 'png'])

    if file_upload is not None:
        _ = main(input_file=file_upload, procedure=procedure_selected)
with tab2:
    run = st.checkbox('Start Camera')

    if run:
        file_upload = st.camera_input('Capture Document', disabled=not run)
        if file_upload is not None:
            pass
          #_ = main(input_file=file_upload, procedure=procedure_selected, image_size=IMAGE_SIZE)
