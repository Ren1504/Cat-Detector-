import numpy as np
import cv2 as cv
import streamlit as st
from PIL import Image

# st.set_option('depreciation.showfileUploaderEncoding',False)

st.write("""# CAT DETECTOR """)

template = Image.open('template.png')
st.image(template)

file = st.file_uploader('Upload a Image file', type = ['jpg','png','jpeg'])


if file is None:
    st.text('Upload an Image file')

else:
    meow = Image.open(file)
    img = np.array(meow)

    st.subheader('Uploaded Image')
    st.image(meow)

    neighbor = st.slider('Min neighbors',0,10,3)
    
    cat_cascade = cv.CascadeClassifier('cat_cascade.xml')

    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY )

    cat  = cat_cascade.detectMultiScale(gray,scaleFactor = 1.1, minNeighbors = neighbor)
    
    for (x,y,w,h) in cat:
        cv.rectangle(img, (x, y), (x + w, y + h), (255,0,0), 2)
        cv.putText(img, 'CAT', (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

    st.subheader('Detected cats in the Image')
    st.image(img)
    st.write('There are {} cat(s) in the image'.format(len(cat)))