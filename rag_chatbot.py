"""
RAG Чат-бот (Retrieval-Augmented Generation)
Проект: Умный ассистент по машинному обучению

Архитектура:
1. Векторная база данных (ChromaDB) хранит документы с эмбеддингами
2. При запросе ищутся k ближайших соседей по косинусному сходству
3. Найденные документы передаются в LLM как контекст
4. LLM генерирует ответ строго по контексту (без галлюцинаций)
"""

import chromadb
from chromadb.utils import embedding_functions
import ollama

# ============================================================
# ЧАСТЬ 1: ВЕКТОРНАЯ БАЗА ДАННЫХ (ChromaDB)
# ============================================================

print("=" * 60)
print("Инициализация векторной базы данных...")
print("=" * 60)

# 1.1. Создаем клиент базы данных (данные сохраняются на диск)
client = chromadb.PersistentClient(path="./chroma_db")

# 1.2. Функция для автоматического создания эмбеддингов
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# 1.3. Создаем или получаем коллекцию
collection = client.get_or_create_collection(
    name="ml_knowledge_base",
    embedding_function=embedding_func
)

print(f"Коллекция '{collection.name}' готова")
print(f"В базе сейчас {collection.count()} документов")

# 1.4. Если коллекция пустая, добавляем тестовые данные
if collection.count() == 0:
    print("\nДобавляем тестовые документы в базу...")
    
    documents = [
        "Машинное обучение (ML) — это подраздел искусственного интеллекта, изучающий методы построения алгоритмов, способных обучаться на данных.",
        "Градиентный спуск — это алгоритм оптимизации, который шаг за шагом изменяет веса модели для минимизации функции потерь.",
        "Функция потерь (Loss Function) измеряет, насколько предсказания модели отличаются от реальных значений. Примеры: MSE для регрессии, Cross-Entropy для классификации.",
        "Learning Rate (скорость обучения) определяет размер шага при обновлении весов. Слишком большой LR приводит к нестабильности, слишком маленький — к медленному обучению.",
        "Переобучение (Overfitting) возникает, когда модель запоминает шум в обучающих данных вместо общих закономерностей. Признаки: высокий accuracy на train, низкий на test.",
        "Методы борьбы с переобучением: регуляризация (L1, L2), Dropout, Early Stopping, Data Augmentation, упрощение модели.",
        "Precision (точность) = TP / (TP + FP). Показывает, какая доля предсказанных положительных случаев действительно положительна. Важна, когда ложные срабатывания дороги.",
        "Recall (полнота) = TP / (TP + FN). Показывает, какую долю реальных положительных случаев модель нашла. Важна, когда пропуск положительного случая критичен (например, в медицине).",
        "F1-Score — гармоническое среднее Precision и Recall. Используется при несбалансированных классах.",
        "CatBoost — библиотека градиентного бустинга от Яндекса. Главное преимущество: автоматическая работа с категориальными признаками через Target Encoding.",
        "Target Encoding заменяет категорию на среднее значение целевой переменной для этой категории. Например, если 70% клиентов из Москвы вернули кредит, категория 'Москва' заменяется на 0.7.",
        "Нейронная сеть состоит из слоёв нейронов. Каждый нейрон вычисляет взвешенную сумму входов и применяет функцию активации.",
        "Функции активации (ReLU, Sigmoid, Tanh) добавляют нелинейность в нейросеть. Без них многослойная сеть эквивалентна одному слою.",
        "ReLU (Rectified Linear Unit) = max(0, x). Самая популярная функция активации для скрытых слоёв. Не страдает от проблемы затухающего градиента.",
        "Sigmoid сжимает значения в диапазон [0, 1]. Используется для бинарной классификации на выходном слое.",
        "Softmax преобразует выходы в вероятности (сумма = 1). Используется для многоклассовой классификации.",
        "Проблема затухающего градиента (Vanishing Gradient): при обратном распространении градиент экспоненциально уменьшается в глубоких сетях с Sigmoid/Tanh. ReLU решает эту проблему.",
        "BERT (Bidirectional Encoder Representations from Transformers) — модель для понимания текста. Использует механизм Self-Attention для анализа контекста.",
        "Self-Attention позволяет каждому слову 'видеть' все остальные слова в предложении. Вычисляет связи через скалярное произведение Query и Key векторов.",
        "Эмбеддинги (Embeddings) — это векторные представления слов или предложений, кодирующие их смысл. Похожие по смыслу тексты имеют близкие векторы.",
        "Косинусное сходство (Cosine Similarity) измеряет угол между двумя векторами. Значение 1.0 означает идентичность, 0.0 — отсутствие связи.",
        "RAG (Retrieval-Augmented Generation) — архитектура, сочетающая поиск в базе знаний с генерацией ответа LLM. Позволяет избежать галлюцинаций модели.",
        "Векторная база данных (ChromaDB, Qdrant, FAISS) хранит эмбеддинги документов и позволяет быстро находить семантически похожие тексты через k-NN поиск.",
        "Температура (Temperature) в LLM контролирует случайность ответов. Низкая температура (0.1-0.3) для фактов, высокая (0.7-1.0) для креатива."
    ]
    
    metadatas = [
        {"category": "ml_basics", "topic": "definition"},
        {"category": "optimization", "topic": "algorithm"},
        {"category": "optimization", "topic": "concept"},
        {"category": "optimization", "topic": "hyperparameter"},
        {"category": "problems", "topic": "overfitting"},
        {"category": "problems", "topic": "solutions"},
        {"category": "metrics", "topic": "precision"},
        {"category": "metrics", "topic": "recall"},
        {"category": "metrics", "topic": "f1"},
        {"category": "algorithms", "topic": "catboost"},
        {"category": "algorithms", "topic": "target_encoding"},
        {"category": "neural_networks", "topic": "architecture"},
        {"category": "neural_networks", "topic": "activation"},
        {"category": "neural_networks", "topic": "relu"},
        {"category": "neural_networks", "topic": "sigmoid"},
        {"category": "neural_networks", "topic": "softmax"},
        {"category": "neural_networks", "topic": "vanishing_gradient"},
        {"category": "nlp", "topic": "bert"},
        {"category": "nlp", "topic": "attention"},
        {"category": "embeddings", "topic": "concept"},
        {"category": "embeddings", "topic": "cosine_similarity"},
        {"category": "rag", "topic": "architecture"},
        {"category": "rag", "topic": "vector_db"},
        {"category": "rag", "topic": "temperature"}
    ]
    
    ids = [f"doc_{i}" for i in range(len(documents))]
    
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"Добавлено {len(documents)} документов в базу")
else:
    print("База данных уже содержит документы")

print("\nВекторная база данных готова к работе")

# ============================================================
# ЧАСТЬ 2: ПРОВЕРКА OLLAMA
# ============================================================

print("\n" + "=" * 60)
print("Проверка Ollama...")
print("=" * 60)

try:
    # Проверяем, что Ollama запущена
    ollama.list()
    print("Ollama работает!")
    
    # Проверяем, что модель скачана
    models = ollama.list()
    model_names = [m['model'] for m in models['models']]
    
    if 'qwen2.5:1.5b' in model_names:
        print("Модель qwen2.5:1.5b найдена")
    else:
        print("Модель qwen2.5:1.5b не найдена. Скачиваю...")
        print("Это может занять 5-10 минут...")
        ollama.pull('qwen2.5:1.5b')
        print("Модель скачана!")
        
except Exception as e:
    print(f"Ошибка: {e}")
    print("\nУбедись, что:")
    print("1. Ollama установлена (скачай с https://ollama.com/)")
    print("2. Ollama запущена (открой программу)")
    print("3. Модель скачана: ollama pull qwen2.5:1.5b")
    input("\nНажми Enter для выхода...")
    exit()

# ============================================================
# ЧАСТЬ 3: ФУНКЦИЯ ПОИСКА В БАЗЕ
# ============================================================

def search_documents(query, k=2):
    """Ищет k ближайших документов по запросу"""
    results = collection.query(
        query_texts=[query],
        n_results=k
    )
    return results['documents'][0]

# ============================================================
# ЧАСТЬ 4: ФУНКЦИЯ ГЕНЕРАЦИИ ОТВЕТА
# ============================================================

def generate_answer(question, context_documents):
    """Генерирует ответ на основе контекста"""
    
    # Формируем контекст из найденных документов
    context = "\n\n".join(context_documents)
    
    # Системный промпт
    system_prompt = """Ты полезный ассистент для ответов на вопросы по машинному обучению и искусственному интеллекту.

ПРАВИЛА:
1. Отвечай ТОЛЬКО на основе предоставленного контекста
2. Если ответа в контексте нет, так и скажи: "Я не нашел эту информацию в документах"
3. Отвечай на русском языке
4. Будь краток и конкретен
5. Если возможно, приводи примеры"""
    
    # Пользовательский промпт
    user_prompt = f"""КОНТЕКСТ:
{context}

ВОПРОС ПОЛЬЗОВАТЕЛЯ:
{question}"""
    
    # Вызываем Ollama
    response = ollama.chat(
        model='qwen2.5:1.5b',
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ],
        options={
            'temperature': 0.1
        }
    )
    
    return response['message']['content']

# ============================================================
# ЧАСТЬ 5: ИНТЕРАКТИВНЫЙ ЧАТ-БОТ
# ============================================================

print("\n" + "=" * 60)
print("RAG ЧАТ-БОТ ЗАПУЩЕН!")
print("=" * 60)
print("\nПримеры вопросов:")
print("  - Что такое переобучение?")
print("  - Как работает CatBoost?")
print("  - Чем Precision отличается от Recall?")
print("  - Что такое RAG?")
print("  - Как бороться с переобучением?")
print("\nДля выхода напиши: выход, exit или quit")
print("=" * 60)

# Главный цикл чат-бота
while True:
    try:
        # Получаем вопрос от пользователя
        question = input("\nТы: ").strip()
        
        # Проверяем команды выхода
        if question.lower() in ["выход", "exit", "quit", "q"]:
            print("\nДо свидания! Спасибо за использование RAG чат-бота.")
            break
        
        # Пропускаем пустые вопросы
        if not question:
            continue
        
        # Показываем индикатор загрузки
        print("\nДумаю...")
        
        # Шаг 1: Ищем похожие документы в базе
        context_documents = search_documents(question, k=2)
        
        # Шаг 2: Генерируем ответ на основе контекста
        answer = generate_answer(question, context_documents)
        
        # Выводим ответ бота
        print(f"\nАссистент: {answer}")
        
        # Показываем источники
        print("\nИсточники информации:")
        for i, doc in enumerate(context_documents, 1):
            content = doc if len(doc) <= 100 else doc[:100] + "..."
            print(f"  {i}. {content}")
        
        # Разделитель
        print("\n" + "-" * 60)
        
    except KeyboardInterrupt:
        print("\n\nДо свидания!")
        break
    except Exception as e:
        print(f"\nПроизошла ошибка: {e}")
        print("Попробуй задать вопрос ещё раз.")

# ============================================================
# ЗАВЕРШЕНИЕ РАБОТЫ
# ============================================================

print("\n" + "=" * 60)
print("Сессия завершена")
print("=" * 60)
print(f"Всего документов в базе: {collection.count()}")
print(f"Данные сохранены в: ./chroma_db")
print("=" * 60)