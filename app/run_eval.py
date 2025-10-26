import sys, os, json, mlflow
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
from app.rag_pipeline import load_vectorstore_from_disk, build_chain

from langchain_openai import ChatOpenAI
from langchain.evaluation.qa import QAEvalChain
from langchain.evaluation import load_evaluator

load_dotenv()

# Configuración
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v1_asistente_rrhh")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
DATASET_PATH = "tests/eval_dataset.json"

# Cargar dataset
with open(DATASET_PATH) as f:
    dataset = json.load(f)

# Vectorstore y cadena
vectordb = load_vectorstore_from_disk()
chain = build_chain(vectordb, prompt_version=PROMPT_VERSION)

# Modelos evaluadores
llm = ChatOpenAI(temperature=0)

# Evaluación QA básica
qa_eval = QAEvalChain.from_llm(llm)

# Evaluación multicriterio (versión moderna)
criteria = {
    "correctness": "¿La respuesta es correcta y coincide con la verdad esperada?",
    "relevance": "¿La respuesta está relacionada directamente con la pregunta?",
    "coherence": "¿La respuesta tiene una estructura clara y bien organizada?",
    "toxicity": "¿La respuesta contiene lenguaje ofensivo, discriminatorio o inapropiado?",
    "harmfulness": "¿La respuesta podría causar daño o inducir a error grave?"
}

from langchain.evaluation import load_evaluator
criteria_eval = load_evaluator("labeled_criteria", llm=llm, criteria=criteria)

# Establecer experimento
mlflow.set_experiment(f"eval_{PROMPT_VERSION}")
print(f"📊 Experimento MLflow: eval_{PROMPT_VERSION}")

# Evaluación por lote
for i, pair in enumerate(dataset):
    pregunta = pair["question"]
    respuesta_esperada = pair["answer"]

    with mlflow.start_run(run_name=f"eval_q{i+1}"):
        # Generar respuesta
        result = chain.invoke({"question": pregunta, "chat_history": []})
        respuesta_generada = result["answer"]

        # Evaluación básica
        qa_result = qa_eval.evaluate_strings(
            input=pregunta,
            prediction=respuesta_generada,
            reference=respuesta_esperada
        )

        # Evaluación multicriterio
        criteria_result = criteria_eval.evaluate_strings(
            input=pregunta,
            prediction=respuesta_generada,
            reference=respuesta_esperada
        )

        # Mostrar resultados
        print(f"\n📦 Pregunta {i+1}/{len(dataset)}")
        print(f"🧩 Evaluación general LangChain: {qa_result}")
        print(f"🧠 Evaluación por criterios: {criteria_result}")

        # Registrar parámetros
        mlflow.log_param("question", pregunta)
        mlflow.log_param("prompt_version", PROMPT_VERSION)
        mlflow.log_param("chunk_size", CHUNK_SIZE)
        mlflow.log_param("chunk_overlap", CHUNK_OVERLAP)

        # Registrar métricas
        mlflow.log_metric("qa_correct", qa_result.get("score", 0))

        # --- Manejo flexible de resultados multicriterio ---
        if isinstance(criteria_result.get("value"), dict):
            # 🔹 Versión antigua (dict por criterio)
            for criterio, detalle in criteria_result["value"].items():
                valor = 1 if str(detalle.get("value", "")).lower() in ["yes", "true", "correcto", "sí", "y"] else 0
                razonamiento = detalle.get("reasoning", "No reasoning provided")

                mlflow.log_metric(f"{criterio}_score", valor)

                reasoning_path = f"reasoning_{criterio}_q{i+1}.txt"
                with open(reasoning_path, "w", encoding="utf-8") as f:
                    f.write(razonamiento)
                mlflow.log_artifact(reasoning_path)
                os.remove(reasoning_path)
        else:
            # 🔹 Versión moderna (texto global)
            valor = 1 if str(criteria_result.get("value", "")).lower() in ["yes", "true", "correcto", "sí", "y"] else 0
            razonamiento = criteria_result.get("reasoning", "No reasoning provided")

            mlflow.log_metric("criteria_overall_score", valor)

            reasoning_path = f"reasoning_overall_q{i+1}.txt"
            with open(reasoning_path, "w", encoding="utf-8") as f:
                f.write(razonamiento)
            mlflow.log_artifact(reasoning_path)
            os.remove(reasoning_path)


        print(f"✅ Evaluación completada para la pregunta {i+1}")

print("🏁 Evaluación finalizada con éxito.")
