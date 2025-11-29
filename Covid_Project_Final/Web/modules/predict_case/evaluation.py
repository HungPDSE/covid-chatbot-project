import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(model, X_test_seq, X_test_country, y_test):
    """Dự đoán và đánh giá kết quả dựa trên tập test"""
    y_pred = model.predict({
        "sequence_input": X_test_seq,
        "country_input": X_test_country
    })
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Đánh giá mô hình trên tập test:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    return y_pred, mse, mae, r2

def plot_predictions(y_test, y_pred):
    """Plot giá trị thực tế và giá trị dự đoán của model"""
    y_true_plot = y_test.flatten()
    y_pred_plot = y_pred.flatten()
    plt.figure(figsize=(12, 5))
    plt.plot(y_true_plot, label='Thực tế (y_true)', color='blue', linewidth=2)
    plt.plot(y_pred_plot, label='Dự đoán (y_pred)', color='red', linestyle='--', linewidth=2)
    plt.title("So sánh giữa giá trị thực tế và dự đoán")
    plt.xlabel("Mẫu (time step)")
    plt.ylabel("Số ca mắc")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def estimate_accuracy(y_test, y_pred):
    """Ước lượng tỷ lệ giữa dự đoán và thực tế"""
    non_zero = y_test.flatten() != 0
    ratios = y_pred.flatten()[non_zero] / y_test.flatten()[non_zero]
    ratios = ratios[ratios < 10]
    mean_ratio = np.mean(ratios)
    accuracy_est = mean_ratio * 100
    print(f"Trung bình tỉ lệ y_pred / y_true: {mean_ratio:.2f}")
    print(f"Ước lượng mô hình dự đoán đúng khoảng: {accuracy_est:.2f}%")
    return mean_ratio, accuracy_est
