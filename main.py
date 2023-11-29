import streamlit as st
from PIL import Image
import pytesseract
import os
import pandas as pd

print(os.getcwd())
os.environ["TESSDATA_PREFIX"] = "C:/Program Files/Tesseract-OCR/tessdata"

def process_image(img_pro):
    tessdata_dir_config = f'--tessdata-dir "{os.environ["TESSDATA_PREFIX"]}"'
    
    text = pytesseract.image_to_string(img_pro, lang='ind', config=tessdata_dir_config)

    sections = {}
    lines = text.split('\n')
    current_section = ''

    i = 1
    for line in lines:
        if line.strip() == "":
            continue

        if "Name" in line:
            current_section = "section_3"

        elif "Code" in line:
            current_section = "section_6"

        else:
            current_section = f"section_{i}"

        sections[current_section] = line.strip()
        i += 1

    print(sections)
    university = sections.get('section_1', '')
    faculty = sections.get('section_2', '')
    name = sections.get('section_3', '')[7:] + ' ' + sections.get('section_4', '') + ' ' + sections.get('section_5', '')
    id = sections.get('section_6', '')[6:]

    return name, university, faculty, id 

def main():
    st.title("Recogniton ID ERU")

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    image_preview = st.empty()

    if uploaded_image is not None:
        img_pro = Image.open(uploaded_image)
        image_preview.image(img_pro, caption="Uploaded Image", use_column_width=True)

        if st.button("Process Image"):
            name, university, faculty, id = process_image(img_pro)  # Use _ to discard id_image

            data = {
                "Name": [name],
                "University": [university],
                "Faculty": [faculty],
                "ID": [id]
            }
            df = pd.DataFrame(data)

            # Display DataFrame
            st.write("Processed Results:")
            st.dataframe(df)

if __name__ == "__main__":
    main()
