import tensorflow as tf
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

label_encoder = joblib.load("Web/data/label_encoder.pkl")


# Đường dẫn tới model và label encoder đã training
model_path = Path("Web/data/bilstm_covid19_model_with_emb.h5")
encoder_path = Path("Web/data/label_encoder.pkl")

def get_country_embeddings():
    print("Đang tải model...")
    model = tf.keras.models.load_model(str(model_path))
    print("Model đã tải thành công.")

    # Tìm embedding layer
    embedding_layer = None
    for layer in model.layers:
        if "embedding" in layer.name.lower() and isinstance(layer, tf.keras.layers.Embedding):
            embedding_layer = layer
            break

    if embedding_layer is None:
        print("Không tìm thấy Embedding Layer trong model.")
        return

    embeddings = embedding_layer.get_weights()[0]
    print(f"Kích thước ma trận embedding: {embeddings.shape}")

    # Load LabelEncoder đã được training
    if not encoder_path.exists():
        print(f"Không tìm thấy file LabelEncoder tại {encoder_path}")
        return

    with open(encoder_path, "rb") as f:
        label_encoder = joblib.load(f)

    print(f"Đã load LabelEncoder với {len(label_encoder.classes_)} quốc gia.")

    # Mapping index → country name theo đúng thứ tự cũ
    country_embeddings = {}
    for idx in range(min(len(label_encoder.classes_), embeddings.shape[0])):
        country_name = label_encoder.inverse_transform([idx])[0]
        country_embeddings[country_name] = embeddings[idx]

    print(f"Đã trích xuất embedding cho {len(country_embeddings)} quốc gia.")

    # Lưu vào file để dùng sau
    with open("country_embeddings.pkl", "wb") as f:
        pickle.dump(country_embeddings, f)
    print("Đã lưu vào country_embeddings.pkl")

    # In thử vài embedding
    print("\nMột vài ví dụ:")
    for i, (country, emb) in enumerate(country_embeddings.items()):
        if i < 5:
            print(f"{country}: {emb[:5]}...")
        else:
            break

if __name__ == "__main__":
    get_country_embeddings()
