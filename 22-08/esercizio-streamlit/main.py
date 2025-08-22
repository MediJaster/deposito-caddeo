from numpy import size
import streamlit as st

"""
# Counter app

A simple counter app using Streamlit.

"""


if "count" not in st.session_state:
    st.session_state.count = 0

f"### Current count: {st.session_state.count}"

col1, col2, col3 = st.columns(3, gap="small")

with col1:
    if st.button("Decrement"):
        st.session_state.count -= 1

with col2:
    if st.button("Reset"):
        st.session_state.count = 0

with col3:
    if st.button("Increment"):
        st.session_state.count += 1
