import dotenv
import os
from pathlib import Path

import tensorflow as tf
from keras import layers, Model
from keras.preprocessing import image
import numpy as np
import os
import random
import keras
from keras.models import load_model

# -----------------------------
# PATH LOADING FUNCTIONS
# -----------------------------

def load_env(env_path:str="paths.env") -> dict[str, str] | None:
    path = os.path.abspath(env_path)
    
    if Path(path).exists():
        return dotenv.dotenv_values(os.path.abspath(env_path))
    
    else:
        prompt = f"Did not find environment file '{path or env_path}'.\nDo you want to generate it? [y/n]\n"
        response = input(prompt).strip().lower()
        if response not in ['y', 'yes']:
            return None
        
        env = {
            "ROOT_DIR": os.getcwd(),
            "DATASET_DIR": "UBIPeriocular",
            "SAVE_PATH": "siamese_eye_model.keras",
            "GALLERY_DIR": "Gallery",
            "QUERY_IMAGE_PATH": "query_image.jpg",
        }

        with open(env_path, 'wt') as f:
            for (k,v) in env.items():
                f.write(k + "=" + v + "\n")
            f.flush()

        return env


def build_paths(env: dict[str, str]) -> tuple[Path, Path, Path, Path]:
    
    for key in ["DATASET_DIR", "SAVE_PATH", "GALLERY_DIR", "QUERY_IMAGE_PATH"]:
        value = env.get(key)
        if value is None:
            raise Exception(f"env variable '{key}' not found")

    if "ROOT_DIR" in env:
        root_path = Path(env.get("ROOT_DIR")).expanduser()
    else:
        root_path = os.getcwd()

    dataset = (root_path / Path(env.get("DATASET_DIR"))).resolve()
    save = (root_path / Path(env.get("SAVE_PATH"))).resolve()
    gallery = (root_path / Path(env.get("GALLERY_DIR"))).resolve()
    query_image = (root_path / Path(env.get("QUERY_IMAGE_PATH"))).resolve()

    return (dataset, save, gallery, query_image)



# -----------------------------
# UTILITY FUNCTIONS
# -----------------------------
def load_images_by_filename(dataset_path, img_size):
    """
    Load images and group them by person_id extracted from filename (e.g., C1_S1).
    """
    images_dict = {}
    for img_name in os.listdir(dataset_path):
        if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
            person_id = "_".join(img_name.split("_")[:2])  # e.g., C1_S1
            img_path = os.path.join(dataset_path, img_name)
            img = image.load_img(img_path, target_size=img_size)
            img = image.img_to_array(img) / 255.0
            if person_id not in images_dict:
                images_dict[person_id] = []
            images_dict[person_id].append(img)
    if not images_dict:
        raise ValueError("No images found in dataset. Check DATASET_PATH and file names.")
    return images_dict

def siamese_batch_generator(images_dict, batch_size, img_size): 
    """ 
    Generate batches of pairs for Siamese network training. 
    Does so dynamically to avoid large memory usage. 
    Adds tiny noise to duplicate images to avoid exact zeros. 
    """ 
    person_ids = list(images_dict.keys()) 
    if not person_ids: 
        raise ValueError("No persons found in images_dict.") 
    while True: 
        X1 = np.zeros((batch_size, *img_size, 3), dtype=np.float32) 
        X2 = np.zeros((batch_size, *img_size, 3), dtype=np.float32) 
        y = np.zeros((batch_size,), dtype=np.float32) 
        for i in range(batch_size): 
            if random.random() < 0.5: # same class 
                person = random.choice(person_ids) 
                imgs = images_dict[person] 
                if len(imgs) < 2: 
                    img1 = imgs[0] + np.random.normal(0, 1e-3, size=imgs[0].shape) 
                    img2 = imgs[0] + np.random.normal(0, 1e-3, size=imgs[0].shape) 
                else: 
                    img1, img2 = random.sample(imgs, 2) 
                label = 1 
            else: # different class 
                if len(person_ids) > 1: 
                    person1, person2 = random.sample(person_ids, 2) 
                    img1 = random.choice(images_dict[person1]) 
                    img2 = random.choice(images_dict[person2]) 
                    label = 0 
                else: # Only one person, duplicate image with tiny noise 
                    imgs = images_dict[person_ids[0]] 
                    img1 = imgs[0] + np.random.normal(0, 1e-3, size=imgs[0].shape) 
                    img2 = imgs[0] + np.random.normal(0, 1e-3, size=imgs[0].shape) 
                    label = 0.1 
            X1[i] = img1 
            X2[i] = img2 
            y[i] = label 
        yield (X1, X2), y


def make_tf_dataset(images_dict, batch_size, img_size):
    """
    Wrap the generator in a tf.data.Dataset with proper output_signature.
    """
    def gen():
        for batch in siamese_batch_generator(images_dict, batch_size, img_size):
            yield batch
    
    output_signature = (
        (
            tf.TensorSpec(shape=(batch_size, *img_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(batch_size, *img_size, 3), dtype=tf.float32)
        ),
        tf.TensorSpec(shape=(batch_size,), dtype=tf.float32)
    )
    
    dataset = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    return dataset




# -----------------------------
# SIAMESE NETWORK MODEL
# -----------------------------
def create_base_cnn(input_shape):
    """
    Base CNN to extract features from each image.
    """
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, (10,10), activation='relu')(inp)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, (7,7), activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, (4,4), activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(256, (4,4), activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation='sigmoid')(x)
    return Model(inp, x)

@keras.saving.register_keras_serializable()
def euclidean_distance(vects):
    """
    Calculatees similarity in features of 2 images in terms of euclidean distance.
    """
    x, y = vects
    return tf.sqrt(tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True))

def create_siamese_model(input_shape):
    """
    Creates a model that takes 2 input images, extracts their features, and outputs the distance between them.
    """
    base_model = create_base_cnn(input_shape)
    input_a = layers.Input(shape=input_shape)
    input_b = layers.Input(shape=input_shape)
    feat_a = base_model(input_a)
    feat_b = base_model(input_b)
    distance = layers.Lambda(euclidean_distance)([feat_a, feat_b])
    model = Model([input_a, input_b], distance)
    return model

# -----------------------------
# CONTRASTIVE LOSS
# -----------------------------
def contrastive_loss(y_true, y_pred, margin=1.0):
    """
    Defines a loss metric with 2 goals:
    1. Distance small for same person.
    2. Distance large for different people.
    This loss function penalizes the model when either of the above goals are not achieved.
    """
    y_true = tf.cast(y_true, y_pred.dtype)
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)

# -----------------------------
# ACCURACY METRIC
# -----------------------------
@keras.saving.register_keras_serializable()
def siamese_accuracy(y_true, y_pred, threshold=0.5):
    """
    Definition of accuracy for siamese model.
    """
    y_pred_binary = tf.cast(y_pred < threshold, tf.float32)
    return tf.keras.metrics.binary_accuracy(y_true, y_pred_binary)



# -----------------------------
# SAVE MODEL
# -----------------------------
def save_siamese_model(model, save_path="siamese_eye_model"):
    """
    Saves the entire model (structure + weights + optimizer state)
    to a .keras format (recommended).
    """
    model.save(save_path)  # Creates a folder or .keras file
    print(f"Model saved to: {save_path}")


# -----------------------------
# LOAD MODEL
# -----------------------------
def euclidean_distance(vects):
    x, y = vects
    return tf.sqrt(tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True))

def contrastive_loss(y_true, y_pred, margin=1.0):
    y_true = tf.cast(y_true, y_pred.dtype)
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)

def load_siamese_model(save_path):
    """
    Loads a saved Siamese model that uses custom contrastive loss.
    """
    return load_model(save_path, custom_objects={"contrastive_loss": contrastive_loss})




def load_gallery_embeddings(gallery_root):
    """
    Loads all images from the gallery and computes embeddings.
    Returns a dict: {identity: [image_paths]}
    """
    gallery_dict = {}
    for identity in os.listdir(gallery_root):
        identity_path = os.path.join(gallery_root, identity)
        if os.path.isdir(identity_path):
            gallery_dict[identity] = [os.path.join(identity_path, f)
                                      for f in os.listdir(identity_path)
                                      if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    return gallery_dict

def identify_eye(model, query_img_path, gallery_dict, img_size, margin=1.0, threshold=70.0):
    """
    Identify the identity of a query eye image.
    - model: trained Siamese network
    - query_img_path: path to the query image
    - gallery_dict: {identity: [list of image paths]}
    - margin: used for similarity scaling
    - threshold: minimum similarity (%) to accept as known identity
    """
    # Load and preprocess query image
    query_img = image.load_img(query_img_path, target_size=img_size)
    query_img = image.img_to_array(query_img) / 255.0
    query_img = np.expand_dims(query_img, axis=0)

    identity_scores = {}

    for identity, img_paths in gallery_dict.items():
        similarities = []
        for g_path in img_paths:
            try:
                g_img = image.load_img(g_path, target_size=img_size)
                g_img = image.img_to_array(g_img)/255.0
                g_img = np.expand_dims(g_img, axis=0)
                distance = float(model.predict([query_img, g_img], verbose=0)[0,0])
                similarity = (1 - np.tanh(distance / margin)) * 100
                similarities.append(similarity)
            except Exception as e:
                print(f"Skipping {g_path}: {e}")
        if similarities:
            # Take maximum similarity among images for this identity
            identity_scores[identity] = max(similarities)

    if not identity_scores:
        return "No gallery images found", 0.0

    # Determine the best match
    best_identity = max(identity_scores, key=identity_scores.get)
    best_score = identity_scores[best_identity]

    if best_score >= threshold:
        return best_identity, best_score
    else:
        return "Unknown", best_score
