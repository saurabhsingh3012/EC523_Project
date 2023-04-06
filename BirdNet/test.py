import tensorflow as tf
import config as cfg

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path=cfg.MODEL_PATH)
interpreter.allocate_tensors()

# Try to retrieve the metadata (may raise ValueError)
try:
    metadata = interpreter.get_tensor_details()[0]["quantization_parameters"]
    metadata_extractor = tf.lite.support.metadata.MetadataExtractor.get_metadata_extractor_from_object(metadata)

    # Print the input and output tensor details
    print("Input tensor details:")
    for i in range(metadata_extractor.get_input_tensor_count()):
        input_tensor = metadata_extractor.get_input_tensor(i)
        print(input_tensor.name, input_tensor.shape, input_tensor.dtype)

    print("\nOutput tensor details:")
    for i in range(metadata_extractor.get_output_tensor_count()):
        output_tensor = metadata_extractor.get_output_tensor(i)
        print(output_tensor.name, output_tensor.shape, output_tensor.dtype)

    # Print the metadata
    print("\nMetadata:")
    for name, value in metadata_extractor.get_metadata().items():
        print("{}: {}".format(name, value))

except ValueError:
    print("Model does not contain metadata.")

