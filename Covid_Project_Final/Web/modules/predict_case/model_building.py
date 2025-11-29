from tensorflow.keras import layers, models, Input

def build_bilstm_model(timesteps, num_features, num_countries, emb_dim=18, lstm_units=128, dense_units=64):
    """Xây dựng mô hình Bi-LSTM với embedding cho quốc gia."""
    # Sequence input
    input_seq = Input(shape=(timesteps, num_features), name="sequence_input")
    # Country input (integer id)
    input_country = Input(shape=(1,), name="country_input")
    # Embedding for country
    country_emb = layers.Embedding(input_dim=num_countries, output_dim=emb_dim, name="country_embedding")(input_country)
    country_emb = layers.Flatten()(country_emb)
    # Bi-LSTM branch
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))(input_seq)
    x = layers.Dropout(0.4)(x)
    x = layers.Bidirectional(layers.LSTM(lstm_units))(x)
    x = layers.Dropout(0.4)(x)
    # Concatenate embedding và LSTM output
    x = layers.Concatenate()([x, country_emb])
    # Dense layers
    x = layers.Dense(dense_units, activation="relu")(x)
    outputs = layers.Dense(1, activation="relu")(x)
    # Model
    model = models.Model(inputs=[input_seq, input_country], outputs=outputs)
    model.compile(optimizer="adam", loss="huber", metrics=["mae"])
    return model
