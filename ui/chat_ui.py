import streamlit as st

def chat_ui(chat_engine, collection_name: str):
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    st.subheader('Chat with the video')
    
    # Use a form to prevent rerunning the whole script on text input
    with st.form(key='chat_form', clear_on_submit=True):
        query = st.text_input('Ask something about the video', key='query_input')
        submit_button = st.form_submit_button(label='Ask')

    if submit_button and query:
        with st.spinner('Retrieving answer...'):
            res = chat_engine.answer(collection_name, query, st.session_state.chat_history)
            
            if isinstance(res, dict):
                answer = res.get('answer')
            else: # Fallback for unexpected response format
                answer = str(res)

            st.session_state.chat_history.append((query, answer))
    
    # Display conversation history
    if st.session_state.chat_history:
        st.markdown('---')
        # Display the latest answer immediately below the input
        latest_q, latest_a = st.session_state.chat_history[-1]
        st.markdown(f"**You:** {latest_q}")
        st.markdown(f"**Assistant:** {latest_a}")
