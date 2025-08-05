import streamlit as st
import pandas as pd
import pickle

# App title
st.title("Voice Gender Prediction")

file = st.file_uploader("Upload csv file", type="csv")

# If file is uploaded
if file is not None:
    data = pd.read_csv(file)
    st.write("Uploaded data:")
    st.write(data)

    model_file = open("rf_model.pkl", "rb")
    model = pickle.load(model_file)

    # predict
    result = model.predict(data)

    # show result
    st.write("Prediction (0 = female, 1 = male):")
    st.write(result)

    # in words
    for r in result:
        if r == 0:
            st.write("This is female")
        else:
            st.write("This is male")