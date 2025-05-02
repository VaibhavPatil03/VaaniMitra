import streamlit as st
import base64

# 1. Page Configuration
st.set_page_config(
    page_title="Vaani Mitra",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Import only the functions, not modules
from components.sidebar import sidebar
from components.topbar import topbar
from components.chatbot import show as chatbot_show  # renamed for clarity
from utils.auth import check_login_state
from pages import (
    home,
    login,
    register,
    dashboard,
    object_detection_page,
    sign_language_page,
    misarticulation_page
)

# 3. Custom Styling: Hide Streamlit default formatting
hide_default_format = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""

# 4. Set background image using base64 encoding
def set_background(png_file_path):
    with open(png_file_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Inject Custom CSS and Background
st.markdown(hide_default_format, unsafe_allow_html=True)
set_background("bg3.jpg")  # Make sure the image is in the root directory

# 5. Sidebar (returns selected page)
selected_page = sidebar()

# 6. Topbar
topbar()

# 7. Page Navigation
if not check_login_state():
    if selected_page == "Register":
        register.show()
    else:
        login.show()
else:
    if selected_page == "Home":
        home.show()
    elif selected_page == "Dashboard":
        dashboard.show()
    elif selected_page == "Object Detection":
        object_detection_page.show()
    elif selected_page == "Sign Language Translator":
        sign_language_page.show()
    elif selected_page == "Misarticulation Therapy":
        misarticulation_page.show()
    elif selected_page == "Chatbot":
        chatbot_show()
    elif selected_page == "Logout":  # ✅ Handle Logout
        st.session_state.logged_in = False
        st.session_state.user_details = None
        st.success("✅ You have been logged out.")
        st.rerun()
