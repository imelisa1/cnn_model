def train_model(model, x_train, y_train, x_validation, y_validation, epochs=10, batch_size=64):
    model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_validation, y_validation)
    )
