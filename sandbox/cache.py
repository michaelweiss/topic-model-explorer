import streamlit as st
import numpy as np 
import pandas as pd 
import time

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def expensive_calculation(a, b):
	st.write("Cache miss: expensive_calculation")
	time.sleep(2)
	return { 'output': a * b }  # mutable

a = st.slider("a", 1, 5, 1)
b = st.slider("b", 1, 5, 1)

r = expensive_calculation(a, b)

st.write(r)

