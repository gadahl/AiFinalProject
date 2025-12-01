import dotenv
import os
from pathlib import Path


def load_env(env_path:str="paths.env") -> dict[str, str] | None:
    env = None
    path = None

    try:
        path = os.path.abspath(env_path)
        env = dotenv.dotenv_values(os.path.abspath(env_path))

    except Exception as e:
        print(f"Did not find or could not open environment file '{path or env_path}'.")
        if input("Do you want to generate it? [Y/n]").strip().lower() == 'n':
            return None
        
        env = {
            "ROOT_DIR": os.getcwd(),
            "DATASET_DIR": "UBIPeriocular",
            "SAVE_PATH": "siamese_eye_model.keras",
            "GALLERY_DIR": "gallery",
            "QUERY_IMAGE_PATH": "query_image.png",
        }

        with open(env_path, 'wt') as f:
            for (k,v) in env:
                f.write(k + "=" + v + "\n")
    
    return env



def build_paths(env: dict[str, str]) -> tuple[Path, Path, Path, Path]:
    
    for key in ["ROOT_DIR", "DATASET_DIR", "SAVE_PATH", "GALLERY_DIR", "QUERY_IMAGE_PATH"]:
        value = env.get(key)
        if value is None:
            raise Exception(f"env variable '{key}' not found")

    root_path = Path(env.get("ROOT_DIR")).expanduser()

    dataset = (root_path / Path(env.get("DATASET_DIR"))).resolve()
    save = (root_path / Path(env.get("SAVE_PATH"))).resolve()
    gallery = (root_path / Path(env.get("GALLERY_DIR"))).resolve()
    query_image = (root_path / Path(env.get("QUERY_IMAGE_PATH"))).resolve()

    return (dataset, save, gallery, query_image)
