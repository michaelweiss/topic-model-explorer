import streamlit as st
import numpy as np 
import pandas as pd 
import time

if st.sidebar.checkbox('Show table'):
	"""
	# My first app
	First attempt at using data to create a table
	"""
	df = pd.DataFrame({
		'x': [1, 2, 3],
		'x^3': [x**3 for x in [1, 2, 3]]
		})
	df

if st.sidebar.checkbox('Show data frame'):
	"""
	# Line chart
	"""
	x = np.random.normal(0, 3, 60)
	x = x.reshape(20, 3) # 20 rows with 3 cols
	chart_data = pd.DataFrame(
		x,
		columns = ['a', 'b', 'c'])
	st.line_chart(chart_data)

latest_iteration = st.empty()
bar = st.progress(0)

for i in range(100):
	latest_iteration.markdown(f'Iteration {i+1}')
	bar.progress(i+1)
	time.sleep(0.1)

'... and now we are done!'