import mlflow
import mlflow.tensorflow
import lib
import tensorflow as tf

with mlflow.start_run():
    # Load data
    (x_train, y_train), (x_test, y_test) = lib.load_data()

    # Create model
    model = lib.create_model()

    # Compile model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Train model
    history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

    # Log parameters and results
    mlflow.log_param("epochs", 5)
    mlflow.log_metric("train_accuracy", history.history['accuracy'][-1])
    mlflow.log_metric("val_accuracy", history.history['val_accuracy'][-1])

    # Assuming x_test is your test dataset and model is your trained model
    predictions = model.predict(x_test)

    # Infer the signature
    signature = mlflow.models.infer_signature(x_test, predictions)

    # Log the model with the signature
    mlflow.tensorflow.log_model(model, artifact_path="model", signature=signature)