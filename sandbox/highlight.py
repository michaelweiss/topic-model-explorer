import streamlit as st

import pandas as pd
import numpy as np

df = pd.DataFrame([[0, 2, 3], [0, 4, 1], [10, 20, 30]],
	index=[4, 5, 6], columns=['0', '1', '2'])

st.markdown("Data frame")
st.dataframe(df)

st.markdown("Value of element [4, '1']")
st.write(df.loc[4].at['1'])

st.markdown("Row with index 4")
st.dataframe(df.loc[4])

st.markdown("Column with label '1'")
st.dataframe(df['1'])

def highlight_cell_integer(x, row, col, color="lightgreen"):
    '''
    highlight a specific cell
    '''
    color = "background-color: %s" % (color)
    df = pd.DataFrame('', x.index, x.columns)
    df.iloc[row, col] = color
    return df

st.markdown("Highlight column elements (index by integer positions)")
st.dataframe(df.style
	.apply(highlight_cell_integer, row=1, col=2, axis=None)
	.apply(highlight_cell_integer, row=2, col=0, color="lightblue", axis=None))

def highlight_cell_label(x, row, col, color="lightgreen"):
    '''
    highlight a specific cell
    '''
    color = "background-color: %s" % (color)
    df = pd.DataFrame('', x.index, x.columns)
    df[col].loc[row] = color
    return df

st.markdown("Highlight column elements (index by labels)")
st.dataframe(df.style
	.apply(highlight_cell_label, row=5, col='2', axis=None)
	.apply(highlight_cell_label, row=6, col='0', color="lightblue", axis=None))
