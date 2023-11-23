import os
import cv2
import numpy as np
import onnxruntime
from YOLOseg import YOLOseg
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
    layout="centered",  # centered, wide
    # initial_sidebar_state="expanded",
    menu_items={'About': '### SantonioTheFirst',},
)


@st.cache
def load_model(model_path, conf_thres=0.7, iou_thres=0.3):
    return YOLOseg(model_path, conf_thres, iou_thres)


model = YOLOseg(model_path, conf_thres=conf_thres, iou_thres=iou_thres)


def main(input_file, procedure, image_size=640):
    file_bytes = np.asarray(bytearray(input_file.read()), dtype=np.uint8)  # Read bytes


    col1, col2 = st.columns((1, 1))
    with col1:
        st.title("Input")
        st.image(image, channels="RGB", use_column_width=True)

    with col2:
        st.title("Scanned")
        if procedure == "Traditional":
            pass
        else:
          pass
            # model = model_mbv3 if model_selected == "MobilenetV3-Large" else model_r50
            # output = deep_learning_scan(og_image=image, trained_model=model, image_size=image_size)

            # st.image(output, channels="RGB", use_column_width=True)

    # if output is not None:
        # st.markdown(get_image_download_link(output, f"scanned_{input_file.name}", "Download scanned File"), unsafe_allow_html=Tr

'''
# Document scanner
'''

procedure_selected = st.radio("Select Scanning Procedure:", ('Traditional', 'Deep Learning'), index=1, horizontal=True)

tab1, tab2 = st.tabs(['Upload a Document', 'Capture Document'])

with tab1:
    file_upload = st.file_uploader('Upload Document Image:', type=['jpg', 'jpeg', 'png'])

    if file_upload is not None:
        st.write('Uploaded')
        #_ = main(input_file=file_upload, procedure=procedure_selected, image_size=IMAGE_SIZE)
with tab2:
    run = st.checkbox("Start Camera")

    if run:
        file_upload = st.camera_input("Capture Document", disabled=not run)
        if file_upload is not None:
            pass
          #_ = main(input_file=file_upload, procedure=procedure_selected, image_size=IMAGE_SIZE)
