import argparse
import tensorflow as tf


def main(source_path, destination_path):
    # Load your Keras model
    model = tf.keras.models.load_model(source_path)

    # Save it in the TensorFlow SavedModel format
    model.save(destination_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load a Keras model and save it in TensorFlow SavedModel format."
    )
    parser.add_argument(
        "source", type=str, help="Path to the Keras model file (e.g., your_model.h5)"
    )
    parser.add_argument(
        "destination",
        type=str,
        help="Path to save the TensorFlow SavedModel (e.g., saved_model/my_model)",
    )

    args = parser.parse_args()
    main(args.source, args.destination)
