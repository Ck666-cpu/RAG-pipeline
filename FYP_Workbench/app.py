# FYP_Workbench/app.py
import streamlit as st
import os
import sys

# Add current directory to path so imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from view_model import ChatViewModel
from data_types import ChatMessage

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="FYP RAG Workbench", page_icon="🤖", layout="wide")

# --- CSS STYLING ---
st.markdown("""
<style>
    .stChatMessage { padding: 10px; border-radius: 10px; }
    .stButton button { width: 100%; }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE (MVVM BINDING) ---
if "vm" not in st.session_state:
    # Initialize the ViewModel only once per session
    st.session_state.vm = ChatViewModel()

vm = st.session_state.vm

# --- SIDEBAR (Navigation & Tools) ---
with st.sidebar:
    st.title("🎛️ Control Panel")

    if vm.current_user:
        st.success(f"Logged in as: **{vm.current_user.username}**")
        st.info(f"Role: **{vm.current_user.role}**")

        # --- LOGOUT ---
        if st.button("Logout"):
            vm.logout()
            st.rerun()

        st.divider()

        # --- DOCUMENT UPLOADER ---
        st.subheader("📂 Upload Knowledge")

        # Check permissions purely for UI (ViewModel enforces actual security)
        can_upload = vm.current_user.role in ["Admin", "Staff"]

        if can_upload:
            uploaded_file = st.file_uploader("Choose a text/pdf file", type=["txt", "pdf", "md"])
            is_global = st.checkbox("Make Global (Public)?", disabled=(vm.current_user.role != "Admin"))

            if uploaded_file and st.button("Upload File"):
                # Save to temp disk first (ViewModel expects a path)
                temp_path = os.path.join("temp_uploads", uploaded_file.name)
                os.makedirs("temp_uploads", exist_ok=True)

                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                with st.spinner("Indexing into Vector DB..."):
                    msg = vm.upload_document(temp_path, is_global=is_global)
                    st.success(msg)

                # Cleanup
                os.remove(temp_path)
        else:
            st.warning("Your role cannot upload files.")

    else:
        st.warning("Please Log In to access the system.")

# --- MAIN PAGE ---

st.title("🤖 Enterprise RAG Chatbot")

# 1. LOGIN SCREEN
if not vm.current_user:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.subheader("🔐 User Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login", type="primary"):
            if vm.login(username, password):
                st.toast(f"Welcome back, {username}!", icon="👋")
                st.rerun()
            else:
                st.error("Invalid Username or Password")

        # Helper for testing
        with st.expander("Show Test Credentials"):
            st.write("**Master Admin:** master / 123")
            st.write("**You can register others via Python scripts.**")

# 2. CHAT INTERFACE
else:
    # A. Display Chat History
    for msg in vm.chat_history:
        with st.chat_message(msg.role):
            st.markdown(msg.content)
            # Show sources if available (Evidence)
            if msg.debug_sources:
                with st.expander(f"📚 View {len(msg.debug_sources)} References (Confidence: {msg.confidence:.2f})"):
                    for idx, src in enumerate(msg.debug_sources):
                        st.markdown(f"**{idx + 1}. {src.file_name}** ({src.score:.2f})")
                        st.caption(f"...{src.content_snippet}...")

    # B. Chat Input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # 1. Show User Message
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Generate Assistant Response (Streaming)
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            sources_placeholder = st.empty()

            # This calls the ViewModel generator
            stream = vm.send_message(prompt)

            # Loop to update text in real-time
            for updated_msg in stream:
                # If sources just arrived, show them (optional)
                if updated_msg.debug_sources and not sources_placeholder:
                    # We can show a loading indicator or sources preview here
                    pass

                # Update the text block
                response_placeholder.markdown(updated_msg.content + "▌")

            # Final Polish (Remove cursor)
            response_placeholder.markdown(vm.chat_history[-1].content)

            # Show Source Details Block at the end
            final_msg = vm.chat_history[-1]
            if final_msg.debug_sources:
                with sources_placeholder.expander(f"📚 Used {len(final_msg.debug_sources)} References"):
                    for idx, src in enumerate(final_msg.debug_sources):
                        st.markdown(f"**{idx + 1}. {src.file_name}**")
                        st.caption(src.content_snippet)