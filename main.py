import mlflow
import mlflow.tensorflow
import lib

with mlflow.start_run():
    # Load data
    ds_train, ds_test = lib.load_data()

    # Create model
    model = lib.create_model()

    # Train model
    history = model.fit(
        ds_train,
        epochs=6,
        validation_data=ds_test,
    )

    # Log parameters and metrics
    mlflow.log_param("epochs", 6)
    mlflow.log_metric("train_accuracy", history.history['sparse_categorical_accuracy'][-1])
    mlflow.log_metric("val_accuracy", history.history['val_sparse_categorical_accuracy'][-1])

    # Prepare a sample input for signature inference
    sample_input, _ = next(iter(ds_test.take(1)))
    sample_prediction = model.predict(sample_input)

    # Infer the signature
    signature = mlflow.models.infer_signature(sample_input.numpy(), sample_prediction)

    # Log the model with the signature
    mlflow.tensorflow.log_model(model, artifact_path="model", signature=signature)
