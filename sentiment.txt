import re
import matplotlib.pyplot as plt
import tensorflow as tf
import spacy
from transformers import pipeline
from googletrans import Translator
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Bidirectional, LSTM, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

# Ensure spaCy model is installed
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy 'en_core_web_sm' model...")
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Load sentiment and emotion detection models
try:
    sentiment_pipeline = pipeline("sentiment-analysis")
    emotion_pipeline = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", top_k=1)
except Exception as e:
    print(f"❌ Error loading transformers pipeline: {e}")
    exit()

# Load IMDb dataset
VOCAB_SIZE = 10_000
MAX_LENGTH = 200

try:
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)
    word_index = imdb.get_word_index()
except Exception as e:
    print(f"❌ Error loading IMDb dataset: {e}")
    exit()

# Define max sequence length and pad sequences
X_train = pad_sequences(X_train, maxlen=MAX_LENGTH, padding='pre', truncating='post')
X_test = pad_sequences(X_test, maxlen=MAX_LENGTH, padding='pre', truncating='post')

# Build BiLSTM model with improvements to prevent overfitting
model = Sequential([
    Embedding(input_dim=VOCAB_SIZE + 4, output_dim=16),  # Removed input_length
    Bidirectional(LSTM(32, dropout=0.3, recurrent_dropout=0.3)),
    BatchNormalization(),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# EarlyStopping callback to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train model
try:
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=128,
        validation_data=(X_test, y_test),
        callbacks=[early_stop]
    )
except Exception as e:
    print(f"❌ Error during model training: {e}")
    exit()

# Define aspects
ASPECTS = ["acting", "story", "direction", "cinematography", "music", "dialogue", "characters"]

def preprocess_text(text):
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'[^\w\s]', '', text).lower().strip()
    words = text.split()
    sequence = [(word_index.get(word, 0) + 3) if word_index.get(word, 0) < VOCAB_SIZE else 2 for word in words]
    return sequence

def extract_aspects(text):
    doc = nlp(text)
    extracted_aspects = {aspect: [] for aspect in ASPECTS}

    aspect_keywords = {
        "acting": ["acting", "performance", "actor", "actresses", "actors", "role", "character", "performance", "cast", "play", "portrayal", "showing"],
        "story": ["story", "plot", "narrative", "script", "storyline", "story arc", "writing", "story development", "screenplay"],
        "direction": ["direction", "directing", "director", "vision", "style", "management", "handling", "directional", "filmmaking", "direction style", "director's vision"],
        "cinematography": ["cinematography", "camera", "visuals", "scenes","lighting", "frame", "composition", "visual effects", "shot composition"],
        "music": ["music", "soundtrack", "score", "melody", "soundtrack", "composition", "music score"],
        "dialogue": ["dialogue", "script", "lines", "speech", "conversations", "writing", "dialogue delivery", "verbal exchange", "quotes", "talk"],
        "characters": ["characters", "roles", "acting", "cast", "personality", "behavior", "character arcs", "protagonist", "antagonist"]
    }

    for sentence in doc.sents:
        for aspect, keywords in aspect_keywords.items():
            if any(keyword in sentence.text.lower() for keyword in keywords):
                extracted_aspects[aspect].append(sentence.text)

    return extracted_aspects

def analyze_emotions(translated_text):
    try:
        # Ensure input is a string
        if not isinstance(translated_text, str):
            raise ValueError("Input to emotion_pipeline must be a string.")
        
        # Run the emotion detection model
        emotion_result = emotion_pipeline(translated_text)
        print("Debug: Raw Emotion Output =", emotion_result)  # Log the raw output
        
        # Check if emotion_result is a list and has at least one element
        if isinstance(emotion_result, list) and len(emotion_result) > 0:
            first_result = emotion_result[0]  # Get the first element of the outer list
            
            # Check if the first result is a list and has at least one element
            if isinstance(first_result, list) and len(first_result) > 0:
                first_emotion = first_result[0]  # Get the first element of the inner list
                
                # Check if the first emotion contains 'label' and 'score'
                if isinstance(first_emotion, dict) and 'label' in first_emotion and 'score' in first_emotion:
                    # Extract label and score
                    emotion_label = first_emotion['label']
                    emotion_score = first_emotion['score']
                    return f"Emotion: {emotion_label} (Score: {emotion_score:.2f})"
                else:
                    # Log the unexpected structure and return an informative message
                    print(f"Debug: Unexpected structure in first result: {first_emotion}")
                    return "Unexpected structure in the first result of emotion output."
            else:
                print(f"Debug: First result is not a list or is empty: {first_result}")
                return "Unexpected structure in the first result of emotion output."
        else:
            # Log the empty or non-list result and return an informative message
            print(f"Debug: Empty or non-list emotion_result: {emotion_result}")
            return "No emotions detected or unexpected response format."

    except Exception as e:
        print(f"❌ Emotion analysis error: {e}")
        return "Error in emotion detection"

translator = Translator()

def multilingual_sentiment_analysis(text, target_language='en'):
    try:
        translated_obj = translator.translate(text, dest=target_language)
        translated_text = translated_obj.text
        sentiment_result = sentiment_pipeline(translated_text)[0]
        sentiment_label = sentiment_result.get('label', 'Neutral')
        return translated_text, sentiment_label
    except Exception as e:
        print(f"❌ Translation error: {e}")
        return text, "Translation Failed"
    
def analyze_aspect_sentiment(text):
    extracted_aspects = extract_aspects(text)
    aspect_sentiment = {}

    for aspect, sentences in extracted_aspects.items():
        if sentences:  # Only analyze if there are sentences for the aspect
            aspect_sentiment[aspect] = []
            for sentence in sentences:
                sentiment_result = sentiment_pipeline(sentence)[0]
                sentiment_label = sentiment_result.get('label', 'Neutral')
                aspect_sentiment[aspect].append(sentiment_label)
    
    return aspect_sentiment

def main():
    sample_reviews = [
        "The movie had great acting and cinematography, but the story was boring and slow.",
        "Me encantó la banda sonora y la dirección, pero el diálogo fue mal escrito.",
        "J'adore la performance des acteurs, mais le scénario était médiocre."
    ]

    for review in sample_reviews:
        translated_text, sentiment = multilingual_sentiment_analysis(review)
        emotions = analyze_emotions(translated_text)
        aspect_sentiment = analyze_aspect_sentiment(translated_text)

        print(f"Original Review: {review}")
        print(f"Translated Review: {translated_text}")
        print(f"Sentiment: {sentiment}")
        print(f"Emotion: {emotions}")
        print(f"Aspect-Based Sentiment: {aspect_sentiment}\n")

if __name__ == "__main__":
    main()

# Plot Accuracy and Loss
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o', linestyle='-')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o', linestyle='-')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
for i, acc in enumerate(history.history['accuracy']):
    plt.text(i, acc, f"{acc:.2f}", fontsize=8, verticalalignment='bottom')
for i, val_acc in enumerate(history.history['val_accuracy']):
    plt.text(i, val_acc, f"{val_acc:.2f}", fontsize=8, verticalalignment='bottom')

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', marker='o', linestyle='-')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='o', linestyle='-')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
for i, loss in enumerate(history.history['loss']):
    plt.text(i, loss, f"{loss:.2f}", fontsize=8, verticalalignment='top')
for i, val_loss in enumerate(history.history['val_loss']):
    plt.text(i, val_loss, f"{val_loss:.2f}", fontsize=8, verticalalignment='top')

plt.tight_layout()
plt.show()

# Bar graph for final accuracy and loss
final_train_accuracy = history.history['accuracy'][-1]
final_val_accuracy = history.history['val_accuracy'][-1]
final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]

# Create a bar graph
plt.figure(figsize=(10, 6))
bar_width = 0.35
index = range(2)

# Bar positions
bar1 = [final_train_accuracy, final_train_loss]
bar2 = [final_val_accuracy, final_val_loss]

# Create bars
plt.bar(index, bar1, bar_width, label='Train', color='b')
plt.bar([i + bar_width for i in index], bar2, bar_width, label='Validation', color='r')

# Add labels and title
plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('Final Training and Validation Accuracy and Loss')
plt.xticks([i + bar_width / 2 for i in index], ['Accuracy', 'Loss'])
plt.legend()

# Show the bar graph
plt.tight_layout()
plt.show()