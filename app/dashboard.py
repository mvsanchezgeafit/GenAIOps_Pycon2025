import mlflow
import pandas as pd
import streamlit as st
import altair as alt
from io import StringIO

# --- CONFIGURACIÓN GENERAL ---
st.set_page_config(page_title="📊 Dashboard General de Evaluación", layout="wide")
st.title("🤖 Evaluación Completa del Chatbot por Pregunta")

# --- CARGAR EXPERIMENTOS ---
client = mlflow.tracking.MlflowClient()
experiments = [exp for exp in client.search_experiments() if exp.name.startswith("eval_")]

if not experiments:
    st.warning("⚠️ No se encontraron experimentos de evaluación.")
    st.stop()

exp_names = [exp.name for exp in experiments]
selected_exp_name = st.selectbox("Selecciona un experimento para visualizar:", exp_names)
experiment = client.get_experiment_by_name(selected_exp_name)
runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"])

if not runs:
    st.warning("⚠️ No hay ejecuciones registradas en este experimento.")
    st.stop()

# --- CONVERTIR RUNS A DATAFRAME ---
data = []
for run in runs:
    params = run.data.params
    metrics = run.data.metrics

    # Captura todos los criterios presentes
    row = {
        "run_id": run.info.run_id,
        "pregunta": params.get("question"),
        "prompt_version": params.get("prompt_version"),
        "chunk_size": int(params.get("chunk_size", 0)),
        "chunk_overlap": int(params.get("chunk_overlap", 0)),
    }

    # Agregar métricas dinámicamente
    for k, v in metrics.items():
        row[k] = v

    data.append(row)

df = pd.DataFrame(data)

# --- MOSTRAR TABLA COMPLETA ---
st.subheader("📋 Resultados individuales por pregunta")
st.dataframe(df, use_container_width=True)

# --- ANÁLISIS AGRUPADO ---
metric_cols = [c for c in df.columns if c.endswith("_score") or c == "lc_is_correct"]
agg_dict = {col: "mean" for col in metric_cols}

grouped = df.groupby(["prompt_version", "chunk_size"]).agg(agg_dict)
grouped["preguntas"] = df.groupby(["prompt_version", "chunk_size"])["pregunta"].count()
grouped = grouped.reset_index()

st.subheader("📊 Desempeño agrupado por configuración")
st.dataframe(grouped, use_container_width=True)

# --- GRÁFICO INTERACTIVO ---
st.markdown("### 🎯 Comparación de criterios")

criterio = st.selectbox("Selecciona un criterio a comparar:", metric_cols)
grouped["config"] = grouped["prompt_version"] + " | " + grouped["chunk_size"].astype(str)

chart = (
    alt.Chart(grouped)
    .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
    .encode(
        x=alt.X("config:N", title="Configuración"),
        y=alt.Y(f"{criterio}:Q", title=f"Promedio {criterio}"),
        color=alt.Color("config:N", legend=None),
        tooltip=["prompt_version", "chunk_size", f"{criterio}"]
    )
    .properties(width="container", height=400)
)
st.altair_chart(chart, use_container_width=True)

# --- VER RAZONAMIENTOS OPCIONALES ---
st.markdown("### 🧠 Razonamientos del modelo (opcional)")

selected_run = st.selectbox(
    "Selecciona una ejecución para ver sus razonamientos:",
    df["run_id"].tolist()
)

if selected_run:
    st.info(f"📂 Cargando razonamientos del run: `{selected_run}`")

    artifacts = client.list_artifacts(selected_run)
    reasoning_files = [a for a in artifacts if a.path.endswith(".txt")]

    if reasoning_files:
        for art in reasoning_files:
            file_data = client.download_artifacts(selected_run, art.path)
            content = file_data.read().decode("utf-8") if hasattr(file_data, "read") else open(file_data, "r", encoding="utf-8").read()

            criterio_name = art.path.replace("reasoning_", "").replace(".txt", "")
            with st.expander(f"🧩 {criterio_name.upper()}"):
                st.text_area("Razonamiento", content, height=200)
    else:
        st.warning("⚠️ No se encontraron razonamientos registrados para esta ejecución.")

st.markdown("---")
st.caption("🧠 Desarrollado con LangChain, MLflow y Streamlit – Dashboard extendido de evaluación IA")
