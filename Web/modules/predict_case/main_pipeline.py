from data_processing import load_and_preprocess_data, create_sequences, train_test_split_by_country
from model_building import build_bilstm_model
from training import train_model, plot_training_history
from evaluation import evaluate_model, plot_predictions, estimate_accuracy
from save_utils import save_model, save_test_data

csv_path = "Web\data\Covid19_cleaned_to_model.csv" 
df, le = load_and_preprocess_data(csv_path)

features = [
    'new_cases_log', 'new_deaths_log', 'vaccinations_log',
    'vaccinated_scaled', 'stringency_scaled'
    ]
timesteps = 7
X_seq, X_country, y = create_sequences(df, features, timesteps)

X_train_seq, X_test_seq, X_train_country, X_test_country, y_train, y_test = train_test_split_by_country(X_seq, X_country, y)

num_countries = len(le.classes_)
model = build_bilstm_model(timesteps, len(features), num_countries)
model.summary()

history = train_model(model, X_train_seq, X_train_country, y_train)
plot_training_history(history)

y_pred, mse, mae, r2 = evaluate_model(model, X_test_seq[:20000], X_test_country[:20000], y_test[:20000])
plot_predictions(y_test[:20000], y_pred[:20000])
estimate_accuracy(y_test[:20000], y_pred[:20000])

save_model(model, "data/bilstm_covid19_model_with_emb.h5")
save_test_data(X_test_seq, X_test_country, y_test, le, "data")
