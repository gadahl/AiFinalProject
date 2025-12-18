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


class EnvLoader:
    """
    On creation, members of this class will load path settings from 
    the specified .env file, do some path manipulation, 
    and then save those paths as fields.
    """
    def __init__(self, env_path:str="paths.env"):

        self.defaults = {
                "RAW_IMAGE_DIR": "UBIPeriocular",
                "MODEL_SAVE_FILE": "siamese_eye_model.keras",
                "GALLERY_IMAGE_DIR": "Gallery",
                "GALLERY_EMBEDDING_FILE": "gallery_embeddings.npy",
                "QUERY_IMAGE_FILE": "query_image.jpg",
                "TEST_IMAGE_DIR": "TestImages",
                "PROCESSED_IMAGE_FILE": "images_dict.npy"
            }

        abs_env_path = os.path.abspath(env_path)
    
        if Path(abs_env_path).exists():
            env = dotenv.dotenv_values(abs_env_path)
        
        else:
            prompt = f"Did not find environment file '{abs_env_path or env_path}'.\n" 
            prompt += "Do you want to generate it? [y/n]\n"
            response = input(prompt).strip().lower()
            if response not in ['y', 'yes']:
                raise Exception("No environment file")
            
            env = self.defaults.copy()
            env.update({"ROOT_DIR": os.getcwd()})

            with open(env_path, 'wt') as f:
                for (k,v) in env.items():
                    f.write(k + "=" + v + "\n")
                f.flush()
        
        self.build_paths(env)


    def build_paths(self, env: dict[str, str]):
        
        for key in self.defaults.keys():
            value = env.get(key)
            if value is None:
                msg = f"Env variable '{key}' not found. "
                msg += "Try deleting your .env file and re-running the program."
                raise Exception(msg)

        if "ROOT_DIR" in env:
            self.ROOT_DIR = Path(env.get("ROOT_DIR")).expanduser()
        else:
            self.ROOT_DIR = Path(os.getcwd()).resolve()

        self.RAW_IMAGE_PATH = (self.ROOT_DIR / Path(env.get("RAW_IMAGE_DIR"))).resolve()
        self.MODEL_SAVE_PATH = (self.ROOT_DIR / Path(env.get("MODEL_SAVE_FILE"))).resolve()
        self.GALLERY_IMAGE_PATH = (self.ROOT_DIR / Path(env.get("GALLERY_IMAGE_DIR"))).resolve()
        self.GALLERY_EMBEDDING_PATH = (self.ROOT_DIR / Path(env.get("GALLERY_EMBEDDING_FILE"))).resolve()
        self.QUERY_IMAGE_PATH = (self.ROOT_DIR / Path(env.get("QUERY_IMAGE_FILE"))).resolve()
        self.TEST_IMAGE_PATH = (self.ROOT_DIR / Path(env.get("TEST_IMAGE_DIR"))).resolve()
        self.PROCESSED_IMAGE_PATH = (self.ROOT_DIR / Path(env.get("PROCESSED_IMAGE_FILE"))).resolve()
    

    def __str__(self):
        return f"""EnvLoader[
          RAW_IMAGE_PATH={self.RAW_IMAGE_PATH}
         MODEL_SAVE_PATH={self.MODEL_SAVE_PATH}
      GALLERY_IMAGE_PATH={self.GALLERY_IMAGE_PATH}
  GALLERY_EMBEDDING_PATH={self.GALLERY_EMBEDDING_PATH}
        QUERY_IMAGE_PATH={self.QUERY_IMAGE_PATH}
         TEST_IMAGE_PATH={self.TEST_IMAGE_PATH}
    PROCESSED_IMAGE_PATH={self.PROCESSED_IMAGE_PATH}
]"""


# -----------------------------
# UTILITY FUNCTIONS
# -----------------------------
def load_images_by_filename(dataset_path, img_size):
    """
    Load images, crop eyes, group them by person_id.
    Ignore images without eyes.
    """
    images_dict = {}
    
    # Load Haar cascade for eyes
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    for img_name in os.listdir(dataset_path):
        if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
            person_id = "_".join(img_name.split("_")[:2])
            img_path = os.path.join(dataset_path, img_name)
            img = image.load_img(img_path)
            img = image.img_to_array(img).astype(np.uint8)  # Keep uint8 for eye detector

            # Detect eye
            cropped = detect_eye(img, eye_cascade)
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
    Calculates similarity in features of 2 images in terms of euclidean distance.
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
def save_siamese_model(model, save_path):
    """
    Saves the entire model (structure + weights + optimizer state)
    to a .keras format (recommended).
    """
    model.save(save_path)  # Creates a folder or .keras file
    print(f"Model saved to: {save_path}")


# -----------------------------
# LOAD MODEL
# -----------------------------
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



# -----------------------------
# GALLERY FUNCTIONS
# -----------------------------
def load_gallery_images(gallery_root):
    """
    Loads all images from the gallery.
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

def load_gallery_embeddings(embedding_cache_path):

    if embedding_cache_path and os.path.exists(embedding_cache_path):
        try:
            embeddings_dict = np.load(embedding_cache_path, allow_pickle=True).item()
            print(f"Loaded cached embeddings from {embedding_cache_path}")
            return embeddings_dict
        except Exception as e:
            print(f"Failed to load cached embeddings: {e}")
    
    print(f"Embeddings do not exist at {embedding_cache_path}")
    return None

def compute_gallery_embeddings(base_cnn, gallery_dict, img_size, embedding_cache_path=None):
    """
    Computes embeddings for all gallery images using the base CNN.
    Stores each embedding together with the image path.
    """
    embeddings_dict = {}
    
    for identity, img_paths in gallery_dict.items():
        embeddings_dict[identity] = []
        for img_path in img_paths:
            try:
                img = image.load_img(img_path, target_size=img_size)
                img_arr = image.img_to_array(img) / 255.0
                img_arr = np.expand_dims(img_arr, axis=0)
                emb = base_cnn.predict(img_arr, verbose=0)[0]
                embeddings_dict[identity].append((emb, img_path))  # <-- store as tuple
            except Exception as e:
                print(f"Skipping {img_path}: {e}")

    if embedding_cache_path:
        np.save(embedding_cache_path, embeddings_dict)
        print(f"Saved embeddings to {embedding_cache_path}")

    return embeddings_dict




# -----------------------------
# Identify query image
# -----------------------------
def identify_eye(query_img_path, img_name, base_cnn, gallery_embeddings, img_size, margin=1.0, threshold=70.0):
    """
    Identify a query image after detecting and cropping eye.
    Displays the query and the closest matching gallery image.
    """
    # Load and preprocess the query image
    img = image.load_img(query_img_path)
    img_arr = image.img_to_array(img).astype(np.uint8)

    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    cropped_query = detect_eye(img_arr, eye_cascade)
    if cropped_query is None:
        display_query = np.clip(tf.image.resize(img_arr, img_size[:2]).numpy(), 0, 255).astype(np.uint8)
        
        plt.figure(figsize=(3, 3))
        plt.imshow(display_query)
        plt.title(img_name + " (No Eye Detected)")
        plt.axis('off')
        plt.show()
        return "No eye detected", 0.0
    
    # Resize and normalize for model
    cropped_resized = tf.image.resize(cropped_query, img_size)
    cropped_input = cropped_resized / 255.0
    cropped_input = np.expand_dims(cropped_input, axis=0)

    # Compute embedding
    query_embedding = base_cnn.predict(cropped_input, verbose=0)[0]

    # Compare with gallery embeddings
    best_score = -1
    best_identity = None
    best_gallery_image = None

    for identity, emb_path_list in gallery_embeddings.items():
        for embedding, path in emb_path_list:
            distance = np.linalg.norm(query_embedding - embedding)
            similarity = (1 - np.tanh(distance / margin)) * 100
            if similarity > best_score:
                best_score = similarity
                best_identity = identity
                best_gallery_image = path  # use exact path stored

    if best_identity is None:
        print("No gallery embeddings found.")
        return "No gallery embeddings found", 0.0

    # Prepare images for display
    display_query = np.clip(cropped_resized.numpy(), 0, 255).astype(np.uint8)
    gallery_img = image.load_img(best_gallery_image, target_size=img_size)
    display_gallery = np.clip(image.img_to_array(gallery_img), 0, 255).astype(np.uint8)

    identity = best_identity if best_score >= threshold else "Unknown"
    result_message = f"Eye in '{img_name}' identified as '{identity}' with confidence {best_score:.2f}%"

    # Display query and closest match
    plt.figure(figsize=(6, 3))
    plt.subplot(1,2,1)
    plt.imshow(display_query)
    plt.title(img_name)
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(display_gallery)
    plt.title(f"Closest Match: {best_identity}")
    plt.axis('off')
    
    plt.figtext(0.5, 0.02, result_message, ha='center', fontsize=10)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.show()

    return identity, best_score






    
def detect_eye(image_array: np.array, eye_cascade: cv2.CascadeClassifier) -> np.array:
    """
    Detects eyes in an image (numpy array).
    Returns cropped eye image as numpy array if found, else None.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(eyes) == 0:
        return None  # No eye detected

    # Take first detected eye
    x, y, w, h = eyes[0]
    cropped_eye = image_array[y:y+h, x:x+w]
    return cropped_eye


def crop_gallery_images(gallery_root, img_size):
    """
    Crop all images in the gallery that contain eyes.
    
    Parameters:
    - gallery_root: path to the gallery folder (each subfolder is an identity)
    - img_size: size to resize cropped images
    
    This function replaces the original images with cropped ones, deletes images with no eyes,
    and appends '_cropped' to filenames to avoid double cropping.
    """
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

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
                cropped = detect_eye(img_arr, eye_cascade)
                
                if cropped is None:
                    # Delete images without eyes
                    os.remove(fpath)
                    print(f"Deleted {fpath} (no eye detected)")
                    continue
                
                # Resize and save cropped image
                cropped_resized = tf.image.resize(cropped, img_size)
                cropped_resized = np.clip(cropped_resized.numpy(), 0, 255).astype(np.uint8)
                
                base, ext = os.path.splitext(fname)
                new_fname = base + "_cropped" + ext
                new_path = os.path.join(identity_path, new_fname)
                
                image.array_to_img(cropped_resized).save(new_path)
                os.remove(fpath)  # Remove original
                print(f"Cropped and saved {new_path}")
            
            except Exception as e:
                print(f"Error processing {fpath}: {e}")
