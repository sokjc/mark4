from model import MyModel

import tensorflow as tf
import tensorflow_datasets as tfds

from omegaconf import DictConfig, OmegaConf
import hydra

def normalize_img(image, label):
      """Normalizes images: `uint8` -> `float32`."""
      return tf.cast(image, tf.float32) / 255., label

@hydra.main(version_base=None, config_path=".", config_name="parameters")
def main(cfg: DictConfig):

    (ds_train, ds_test), ds_info = tfds.load(
        'mnist', 
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True
    )

    ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    model = MyModel(
                    num_layers = cfg.model.nlayers,
                    num_nodes = cfg.model.nodes,
                    num_classes=10
                    )
    
    model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    model.fit(
        ds_train,
        epochs=6,
        validation_data=ds_test,
    )

if __name__ == "__main__":
     main()