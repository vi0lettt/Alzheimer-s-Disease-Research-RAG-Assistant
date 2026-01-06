import os
import subprocess
import sys

# ---------------------------
# --- Настройка папок ---
# ---------------------------
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# ---------------------------
# --- Функции для запуска скриптов ---
# ---------------------------
def run_script(script_name):
    print(f"\n--- Запуск {script_name} ---")
    subprocess.check_call([sys.executable, script_name])

def run_streamlit():
    print("\n--- Запуск Streamlit интерфейса ---")
    subprocess.check_call(["streamlit", "run", "src/app.py"])

# ---------------------------
# --- Главная функция ---
# ---------------------------
if __name__ == "__main__":
    # 0. Проверка и установка зависимостей
    print("--- Установка зависимостей ---")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "src/requirements.txt"])

    # 1. Сбор статей
    run_script("src/01_collect_pubmed_and_biorxiv.py")

    # 2. Очистка текстов и EDA
    run_script("src/02_text_cleaning_and_eda.py")

    # 3. Подготовка чанков для RAG
    run_script("src/03_prepare_chunks_for_rag.py")

    # 4. Запуск Streamlit интерфейса
    run_streamlit()
