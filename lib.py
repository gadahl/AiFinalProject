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
import cv2
import matplotlib.pyplot as plt

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
    Load images, crop eyes, group them by person_id.
    Ignore images without eyes.
    """
    images_dict = {}
    for img_name in os.listdir(dataset_path):
        if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
            person_id = "_".join(img_name.split("_")[:2])
            img_path = os.path.join(dataset_path, img_name)
            img = image.load_img(img_path)
            img = image.img_to_array(img).astype(np.uint8)  # Keep uint8 for eye detector

            # Detect eye
            cropped = detect_eye(img)
            if cropped is None:
                continue  # Skip images without eyes

            # Resize to model input size
            cropped = tf.image.resize(cropped, img_size[:2])
            cropped = cropped / 255.0  # normalize

            if person_id not in images_dict:
                images_dict[person_id] = []
            images_dict[person_id].append(cropped)

    if not images_dict:
        raise ValueError("No images with eyes found in dataset. Check dataset and detection.")

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
    return model, base_model


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

# -----------------------------
# Compute gallery embeddings
# -----------------------------
def compute_gallery_embeddings(base_cnn, gallery_dict, img_size, embedding_cache_path=None):
    """
    Computes embeddings for all gallery images using the base CNN.
    Optionally caches them to disk.

    Args:
        base_cnn: the base CNN model (single-input) for embeddings
        gallery_dict: {identity: [list of image paths]}
        img_size: target size for images
        embedding_cache_path: path to save/load cached embeddings

    Returns:
        embeddings_dict: {identity: [np.array(embedding), ...]}
    """
    embeddings_dict = {}

    # Try loading from cache first
    if embedding_cache_path and os.path.exists(embedding_cache_path):
        try:
            embeddings_dict = np.load(embedding_cache_path, allow_pickle=True).item()
            print(f"Loaded cached embeddings from {embedding_cache_path}")
            return embeddings_dict
        except Exception as e:
            print(f"Failed to load cached embeddings ({embedding_cache_path}), will recompute: {e}")

    for identity, img_paths in gallery_dict.items():
        embeddings_dict[identity] = []
        for img_path in img_paths:
            try:
                img = image.load_img(img_path, target_size=img_size)
                img = image.img_to_array(img) / 255.0
                img = np.expand_dims(img, axis=0)
                embedding = base_cnn.predict(img, verbose=0)[0]
                embeddings_dict[identity].append(embedding)
            except Exception as e:
                print(f"Skipping {img_path}: {e}")

    # Save to cache
    if embedding_cache_path:
        np.save(embedding_cache_path, embeddings_dict)
        print(f"Saved embeddings to {embedding_cache_path}")

    return embeddings_dict


# -----------------------------
# Identify query image
# -----------------------------
def identify_eye(query_img_path, base_cnn, gallery_dict, gallery_embeddings, img_size, margin=1.0, threshold=70.0):
    """
    Identify a query image after detecting and cropping eye.
    """
    # Load and preprocess the query image
    img = image.load_img(query_img_path)
    img = image.img_to_array(img).astype(np.uint8)
    
    # Detect eye
    cropped = detect_eye(img)
    if cropped is None:
        return "No eye detected", 0.0
    
    # Resize and normalize
    cropped = tf.image.resize(cropped, img_size[:2])
    cropped = cropped / 255.0
    cropped_exp = np.expand_dims(cropped, axis=0)

    # Compute embedding
    query_embedding = base_cnn.predict(cropped_exp, verbose=0)[0]

    # Compare with gallery embeddings
    identity_scores = {}
    for identity, embeddings in gallery_embeddings.items():
        if not embeddings:
            continue
        distances = [np.linalg.norm(query_embedding - g_emb) for g_emb in embeddings]
        similarities = [(1 - np.tanh(d / margin)) * 100 for d in distances]
        identity_scores[identity] = max(similarities)

    if not identity_scores:
        return "No gallery embeddings found", 0.0

    # Find best match
    best_identity = max(identity_scores, key=identity_scores.get)
    best_score = identity_scores[best_identity]

    # Optional: display the cropped query image and closest match
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plt.imshow(cropped.numpy())
    plt.title("Query Eye")
    plt.axis('off')

    # Find image corresponding to best match
    match_img_path = gallery_dict[best_identity][0]  # first image of identity
    match_img = image.load_img(match_img_path, target_size=img_size)
    match_img = image.img_to_array(match_img)/255.0
    plt.subplot(1,2,2)
    plt.imshow(match_img)
    plt.title(f"Closest Match: {best_identity}")
    plt.axis('off')
    plt.show()

    if best_score >= threshold:
        return best_identity, best_score
    else:
        return "Unknown", best_score


    
def detect_eye(image_array):
    """
    Detects eyes in an image (numpy array or PIL image).
    Returns cropped eye image as numpy array if found, else None.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    # Load Haar cascade for eyes
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(eyes) == 0:
        return None  # No eye detected

    # Take first detected eye
    x, y, w, h = eyes[0]
    cropped_eye = image_array[y:y+h, x:x+w]
    return cropped_eye

import os
from keras.preprocessing import image
import numpy as np
import tensorflow as tf

def crop_gallery_images(gallery_root, detect_eye_fn, img_size=(105, 105)):
    """
    Crop all images in the gallery that contain eyes.
    
    Parameters:
    - gallery_root: path to the gallery folder (each subfolder is an identity)
    - detect_eye_fn: a function that takes a numpy image array and returns the cropped eye or None
    - img_size: size to resize cropped images
    
    This function replaces the original images with cropped ones, deletes images with no eyes,
    and appends '_cropped' to filenames to avoid double cropping.
    """
    for identity in os.listdir(gallery_root):
        identity_path = os.path.join(gallery_root, identity)
        if not os.path.isdir(identity_path):
            continue
        
        for fname in os.listdir(identity_path):
            fpath = os.path.join(identity_path, fname)
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            
            # Skip already cropped images
            if "_cropped" in fname:
                continue
            
            try:
                img = image.load_img(fpath)
                img_arr = image.img_to_array(img).astype(np.uint8)
                cropped = detect_eye_fn(img_arr)
                
                if cropped is None:
                    # Delete images without eyes
                    os.remove(fpath)
                    print(f"Deleted {fpath} (no eye detected)")
                    continue
                
                # Resize and save cropped image
                cropped_resized = tf.image.resize(cropped, img_size[:2])
                cropped_resized = np.clip(cropped_resized.numpy(), 0, 255).astype(np.uint8)
                
                base, ext = os.path.splitext(fname)
                new_fname = base + "_cropped" + ext
                new_path = os.path.join(identity_path, new_fname)
                
                image.array_to_img(cropped_resized).save(new_path)
                os.remove(fpath)  # Remove original
                print(f"Cropped and saved {new_path}")
            
            except Exception as e:
                print(f"Error processing {fpath}: {e}")
