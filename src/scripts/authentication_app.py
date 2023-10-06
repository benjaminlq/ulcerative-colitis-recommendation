import streamlit as st

PASSWORD = "123456"


def handler_verify_key():
    input_password = st.session_state.password_input
    if input_password == PASSWORD:
        st.session_state.password_flag = True
    else:
        st.write("Incorrect Password")


if "password_flag" not in st.session_state:
    st.text_input(
        label="Enter Password",
        key="password_input",
        type="password",
        on_change=handler_verify_key,
        placeholder="Enter Password",
    )

else:
    st.write("Successfully login")
    ## EVERYTHING ELSE HERE ##
