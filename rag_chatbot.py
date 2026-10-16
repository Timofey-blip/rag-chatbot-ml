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
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# ============================================================
# ЧАСТЬ 1: ВЕКТОРНАЯ БАЗА ДАННЫХ (ChromaDB)
# ============================================================

print("=" * 60)
print("Инициализация векторной базы данных...")
print("=" * 60)

# 1.1. Создаем клиент базы данных (данные сохраняются на диск)
client = chromadb.PersistentClient(path="./chroma_db")

# 1.2. Функция для автоматического создания эмбеддингов
# Используем модель all-MiniLM-L6-v2 (быстрая и легкая)
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# 1.3. Создаем или получаем коллекцию
collection = client.get_or_create_collection(
    name="ml_knowledge_base",
    embedding_function=embedding_func
)

print(f"✅ Коллекция '{collection.name}' готова")
print(f"📊 В базе сейчас {collection.count()} документов")

# 1.4. Если коллекция пустая, добавляем тестовые данные
if collection.count() == 0:
    print("\n Добавляем тестовые документы в базу...")
    
    # База знаний по машинному обучению
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
    
    # Метаданные для каждого документа
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
    
    # Уникальные ID для документов
    ids = [f"doc_{i}" for i in range(len(documents))]
    
    # Добавляем всё в базу
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"✅ Добавлено {len(documents)} документов в базу")
else:
    print(" База данных уже содержит документы")

# 1.5. Создаем поисковик (retriever)
# k=2 означает, что будем искать 2 ближайших соседа
retriever = collection.as_retriever(search_kwargs={"k": 2})

print("\n✅ Векторная база данных готова к работе")

# ============================================================
# ЧАСТЬ 2: ПОДКЛЮЧЕНИЕ LLM (Ollama)
# ============================================================

print("\n" + "=" * 60)
print("Подключение языковой модели (LLM)...")
print("=" * 60)

# Инициализируем локальную LLM через Ollama
# Используем Qwen 2.5 (1.5B параметров) — быстрая и качественная
llm = Ollama(model="qwen2.5:1.5b", temperature=0.1)

print("✅ LLM загружена и готова к работе")
print("💡 Модель: Qwen 2.5 (1.5B)")
print("🌡️ Температура: 0.1 (низкая для точных ответов)")

# ============================================================
# ЧАСТЬ 3: СОЗДАНИЕ RAG-ЦЕПОЧКИ
# ============================================================

print("\n" + "=" * 60)
print("Создание RAG-цепочки...")
print("=" * 60)

# Системный промпт — инструкция для LLM
# Этот промпт запрещает модели галлюцинировать
system_prompt = """Ты полезный ассистент для ответов на вопросы по машинному обучению и искусственному интеллекту.

ПРАВИЛА:
1. Отвечай ТОЛЬКО на основе предоставленного контекста
2. Если ответа в контексте нет, так и скажи: "Я не нашел эту информацию в документах"
3. Отвечай на русском языке
4. Будь краток и конкретен
5. Если возможно, приводи примеры

КОНТЕКСТ:
{context}

ВОПРОС ПОЛЬЗОВАТЕЛЯ:
{input}"""

# Создаем шаблон промпта
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
])

# Собираем цепочку: Документы + Промпт → LLM → Ответ
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# Финальная RAG-цепочка: Вопрос → Поиск → Контекст → Ответ
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

print("✅ RAG-система полностью готова!")
print("🔄 Цепочка: Вопрос → Эмбеддинг → Поиск (k-NN) → Контекст → LLM → Ответ")

# ============================================================
# ЧАСТЬ 4: ИНТЕРАКТИВНЫЙ ЧАТ-БОТ
# ============================================================

print("\n" + "=" * 60)
print("🤖 RAG ЧАТ-БОТ ЗАПУЩЕН!")
print("=" * 60)
print("\nПримеры вопросов:")
print("  • Что такое переобучение?")
print("  • Как работает CatBoost?")
print("  • Чем Precision отличается от Recall?")
print("  • Что такое RAG?")
print("  • Как бороться с переобучением?")
print("\nДля выхода напиши: выход, exit или quit")
print("=" * 60)

# Главный цикл чат-бота
while True:
    try:
        # Получаем вопрос от пользователя
        question = input("\n❓ Ты: ").strip()
        
        # Проверяем команды выхода
        if question.lower() in ["выход", "exit", "quit", "q"]:
            print("\n👋 До свидания! Спасибо за использование RAG чат-бота.")
            break
        
        # Пропускаем пустые вопросы
        if not question:
            continue
        
        # Показываем индикатор загрузки
        print("\n🤔 Думаю...")
        
        # Вызываем RAG-цепочку
        result = rag_chain.invoke({"input": question})
        
        # Выводим ответ бота
        print(f"\n🤖 Ассистент: {result['answer']}")
        
        # Показываем источники (найденные документы)
        print("\n📚 Источники информации:")
        for i, doc in enumerate(result['context'], 1):
            # Обрезаем длинные документы для читаемости
            content = doc.page_content
            if len(content) > 100:
                content = content[:100] + "..."
            print(f"  {i}. {content}")
        
        # Разделитель между диалогами
        print("\n" + "-" * 60)
        
    except KeyboardInterrupt:
        # Обработка Ctrl+C
        print("\n\n👋 До свидания!")
        break
    except Exception as e:
        # Обработка ошибок
        print(f"\n❌ Произошла ошибка: {e}")
        print("Попробуй задать вопрос ещё раз.")

# ============================================================
# ЗАВЕРШЕНИЕ РАБОТЫ
# ============================================================

print("\n" + "=" * 60)
print("Сессия завершена")
print("=" * 60)
print(f"📊 Всего документов в базе: {collection.count()}")
print(f"💾 Данные сохранены в: ./chroma_db")
print("=" * 60)