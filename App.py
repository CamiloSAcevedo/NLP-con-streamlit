
import streamlit as st
from transformers import pipeline
import pandas as pd

# --- Configuración de la página ---
st.set_page_config(page_title="Zero-Shot Topic Classifier", layout="centered", page_icon="🧠")

st.title("Clasificador de Tópicos (Zero-Shot)")
st.write(
    "Ingresa un texto y una lista de posibles categorías (separadas por comas). "
    "Usamos `facebook/bart-large-mnli` para clasificar sin reentrenamiento."
)

# --- Cargar el modelo en caché para que se haga solo una vez ---
@st.cache_resource(show_spinner=False)
def cargar_pipeline(model_name: str = "facebook/bart-large-mnli"):
    """Carga y devuelve el pipeline de Hugging Face para zero-shot-classification."""
    pipe = pipeline("zero-shot-classification", model=model_name)
    return pipe

pipe = cargar_pipeline()  # se carga una vez y se reutiliza

# --- Inputs de usuario ---
with st.form("form_classify"):
    text_input = st.text_area("Texto a clasificar", height=180, placeholder="Pegue aquí el texto...")
    labels_input = st.text_input(
        "Etiquetas (separadas por comas)",
        placeholder="ej: deportes, politica, tecnología, salud"
    )
    multi_label = st.checkbox("Permitir múltiples etiquetas (multi-label)", value=False)
    submit = st.form_submit_button("Clasificar")

# --- Validaciones y parseo de etiquetas ---
def parse_labels(s: str):
    # separar por comas, eliminar espacios en blanco, y filtrar vacíos
    labels = [lab.strip() for lab in s.split(",")]
    labels = [lab for lab in labels if lab]
    return labels

if submit:
    if not text_input.strip():
        st.error("Por favor ingresa el texto a analizar.")
    else:
        labels = parse_labels(labels_input)
        if len(labels) == 0:
            st.error("Por favor ingresa al menos una etiqueta (separadas por comas).")
        else:
            # Ejecutar inferencia mostrando un spinner
            with st.spinner("Clasificando..."):
                try:
                    result = pipe(text_input, candidate_labels=labels, multi_label=multi_label)
                except Exception as e:
                    st.error(f"Ocurrió un error al ejecutar el modelo: {e}")
                    result = None

            if result:
                # Estructurar resultados en DataFrame para mostrar y graficar
                # El pipeline puede devolver diccionario con 'labels' y 'scores'
                labels_out = result.get("labels", [])
                scores_out = result.get("scores", [])

                # Convertir a DataFrame y ordenar por score descendente
                df = pd.DataFrame({"label": labels_out, "score": scores_out})
                df = df.sort_values("score", ascending=False).reset_index(drop=True)

                st.subheader("Resultados")
                # Mostrar tabla
                st.table(df.style.format({"score": "{:.3f}"}))

                # Mostrar gráfico de barras (streamlit lo renderiza bonito)
                st.subheader("Puntuaciones por etiqueta")
                # st.bar_chart espera un dataframe con índices, así que lo preparamos
                chart_df = df.set_index("label")
                st.bar_chart(chart_df)

                # Mostrar interpretación simple
                top_label = df.loc[0, "label"]
                top_score = df.loc[0, "score"]
                st.markdown(f"**Etiqueta más probable:** `{top_label}` (score = {top_score:.3f})")

                # Si multi_label -> mostrar umbral sugerido y opción para filtrar
                if multi_label:
                    threshold = st.slider("Umbral para considerar etiqueta como 'presente' (solo multi-label)", 0.0, 1.0, 0.5)
                    present = df[df["score"] >= threshold]
                    if not present.empty:
                        st.markdown("**Etiquetas por encima del umbral:**")
                        for _, row in present.iterrows():
                            st.write(f"- {row['label']} ({row['score']:.3f})")
                    else:
                        st.info("Ninguna etiqueta supera el umbral seleccionado.")
