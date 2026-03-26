import streamlit as st
from PIL import Image

st.set_page_config(page_title="Home")
st.write("# Human Motion Dashboard")

web, paper, tb, challenge = st.columns(4)
web.link_button(
    "Webpage", "http://thor.oru.se", icon="🖥️"
)
#paper.link_button(
#    "Paper", "Todo", icon="📄"
#)


st.markdown(
    """
    Todo

"""
)
layout_img = Image.open("images/logo.png")
st.image(layout_img, use_container_width=True)
