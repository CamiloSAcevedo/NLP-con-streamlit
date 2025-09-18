import streamlit as st
from groq import Groq

# --- Configuración de la página ---
st.set_page_config(page_title="Chatbot con Memoria (Groq)", page_icon="💬", layout="centered")
st.title("🤖 Chatbot Conversacional con Memoria (Groq)")

st.write(
    "Este chatbot usa el modelo **llama3-8b-8192** de Groq. "
    "La conversación se mantiene en memoria de sesión para simular un chat real."
)

# --- Input para API Key (no se guarda en repo ni en secrets) ---
api_key = st.text_input("Introduce tu API Key de Groq:", type="password")

if not api_key:
    st.warning("⚠️ Ingresa tu API Key para empezar a chatear.")
    st.stop()

# --- Inicializar cliente Groq ---
client = Groq(api_key=api_key)

# --- Inicializar historial en memoria ---
if "messages" not in st.session_state:
    st.session_state.messages = []  # lista de diccionarios {role, content}

# --- Mostrar historial previo ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Input de usuario ---
if prompt := st.chat_input("Escribe tu mensaje..."):
    # 1. Guardar mensaje del usuario en historial
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Mostrar en la interfaz
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Enviar historial al modelo Groq
    try:
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=st.session_state.messages,
                )
                reply = response.choices[0].message.content
                st.markdown(reply)

        # 3. Guardar respuesta en historial
        st.session_state.messages.append({"role": "assistant", "content": reply})

    except Exception as e:
        st.error(f"Error al conectar con la API de Groq: {e}")
