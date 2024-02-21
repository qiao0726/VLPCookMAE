import streamlit as st
import pandas as pd
import numpy as np

# Title of the web app
st.title('My First Streamlit App')

# Display a static text
st.write("Here's our first attempt at using data to create a table:")

# Create a random DataFrame and display it
df = pd.DataFrame({
  'first column': np.random.choice(['A', 'B', 'C'], 10),
  'second column': np.random.rand(10)
})
st.write(df)

# Create a line chart
chart_data = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['a', 'b', 'c'])

st.line_chart(chart_data)

# Add interactive widgets, like a slider
x = st.slider('Select a value for x', min_value=0, max_value=100, value=50)
st.write('You selected:', x)

# Use a button to print a message
if st.button('Say hello'):
     st.write('Why hello there')
else:
     st.write('Goodbye')
