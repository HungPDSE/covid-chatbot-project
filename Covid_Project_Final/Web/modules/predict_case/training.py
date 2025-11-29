from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

def train_model(model, X_train_seq, X_train_country, y_train, batch_size=64, epochs=30, val_split=0.1):
    """Huấn luyện model sử dụng thêm earlystopping để dừng huấn luyện khi model đạt điểm tối ưu"""
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    history = model.fit(
        x={
            "sequence_input": X_train_seq,
            "country_input": X_train_country
        },
        y=y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=val_split,
        callbacks=[early_stop],
        verbose=1
    )
    return history

def plot_training_history(history):
    """Plot training và validation loss qua các epochs."""
    plt.figure(figsize=(12, 4))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("Loss theo Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)
    plt.show()
