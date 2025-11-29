import joblib

def save_model(model, model_path):
    """Lưu model keras đã train."""
    model.save(model_path)

def save_test_data(X_test_seq, X_test_country, y_test, le, out_dir):
    """Lưu data test và label encoder vào files."""
    joblib.dump(X_test_seq, f"{out_dir}/X_test_seq.pkl")
    joblib.dump(X_test_country, f"{out_dir}/X_test_country.pkl")
    joblib.dump(y_test, f"{out_dir}/y_test.pkl")
    joblib.dump(le, f"{out_dir}/label_encoder.pkl")
