import base64

import pandas as pd
import streamlit as st

def download_link(dataframe, name):
    csv = dataframe.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = "<a href='data:file/csv;base64,{}' download='{}'>Download</a>".format(b64, name)
    st.markdown(href, unsafe_allow_html=True)

st.header("File Download")

data = [(1, 2, 3)]
df = pd.DataFrame(data, columns=["Col1", "Col2", "Col3"])

download_link(df, 'data.csv')
st.dataframe(df)