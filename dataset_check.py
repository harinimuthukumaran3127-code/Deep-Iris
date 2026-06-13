import tensorflow as tf
print("Program started")

img_height = 224
img_width = 224
batch_size = 32

dataset_dir = "dataset"

train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
print("Dataset loaded successfully")
for images, labels in train_ds:
    print(images.shape)
    print(labels.shape)
    break

val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

print("Classes:", train_ds.class_names)
print("Dataset loaded successfully")
