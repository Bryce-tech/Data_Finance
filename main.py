from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class Modele_classification_chats_chiens(ABC):
    """
    Contrat minimal pour un modele binaire chats/chiens.

    Convention:
      - 0 = chien
      - 1 = chat
    """

    @abstractmethod
    def predire_label_de_cette_image(self, path_image: str) -> int:
        """
        Doit retourner:
          - 0 si chien
          - 1 si chat
        """
        raise NotImplementedError


# /NOTE\ : on suppose que les donnees sont dans le dossier data et a l'interieur de chaque dossier,
#          il y a uniquement des images. Ces quatre path ne sont pas a modifier !
PATH_TRAIN_CHIEN = "data/train/chiens"
PATH_TRAIN_CHAT = "data/train/chats"

PATH_VAL_CHIEN = "data/val/chiens"
PATH_VAL_CHAT = "data/val/chats"


# VOTRE CODE ICI
LABEL_CHIEN = 0
LABEL_CHAT = 1
CLASSES = {LABEL_CHIEN: "chien", LABEL_CHAT: "chat"}

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent.parent
MODEL_PATH_AUGMENTATION = str(ROOT_DIR / "model_chats_chiens_avec_augmentation.keras")

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42
EPOCHS = 8


def lister_images(path_dossier: str) -> List[str]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    dossier = Path(path_dossier)
    if not dossier.is_absolute():
        dossier = BASE_DIR / dossier

    return sorted(
        str(path)
        for path in dossier.glob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


def charger_paths_et_labels(
    paths_chiens: Sequence[str],
    paths_chats: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    image_paths = list(paths_chiens) + list(paths_chats)
    labels = [LABEL_CHIEN] * len(paths_chiens) + [LABEL_CHAT] * len(paths_chats)
    return np.array(image_paths), np.array(labels, dtype=np.int32)


def charger_data_train_val() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_paths, y_train = charger_paths_et_labels(
        lister_images(PATH_TRAIN_CHIEN),
        lister_images(PATH_TRAIN_CHAT),
    )
    val_paths, y_val = charger_paths_et_labels(
        lister_images(PATH_VAL_CHIEN),
        lister_images(PATH_VAL_CHAT),
    )

    if len(train_paths) == 0 or len(val_paths) == 0:
        raise ValueError("Aucune image trouvee dans data/train ou data/val.")

    return train_paths, y_train, val_paths, y_val


def image_vers_vecteur_rgb_normalise(
    path_image: str,
    image_size: Tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
    """Transforme une image en vecteur (hauteur * largeur, 3), normalise entre 0 et 1."""
    image = tf.keras.utils.load_img(path_image, target_size=image_size)
    image_array = tf.keras.utils.img_to_array(image).astype("float32") / 255.0
    return image_array.reshape((-1, 3))


def decoder_image(path_image: tf.Tensor, label: tf.Tensor, image_size: Tuple[int, int]):
    image = tf.io.read_file(path_image)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image, tf.cast(label, tf.float32)


def creer_dataset(
    image_paths: Sequence[str],
    labels: Sequence[int],
    image_size: Tuple[int, int] = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = True,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((list(image_paths), list(labels)))
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(image_paths),
            seed=SEED,
            reshuffle_each_iteration=True,
        )
    dataset = dataset.map(
        lambda path, label: decoder_image(path, label, image_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def choisir_nombre_blocs_convolution(image_size: Tuple[int, int]) -> int:
    nb_pixels = image_size[0] * image_size[1]
    if nb_pixels <= 64 * 64:
        return 2
    if nb_pixels <= 128 * 128:
        return 3
    return 4


def construire_cnn(image_size: Tuple[int, int] = IMAGE_SIZE) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(image_size[0], image_size[1], 3))

    x = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=SEED),
            tf.keras.layers.RandomRotation(0.08, seed=SEED),
            tf.keras.layers.RandomZoom(0.10, seed=SEED),
            tf.keras.layers.RandomContrast(0.10, seed=SEED),
        ],
        name="augmentation_artificielle",
    )(inputs)

    nb_blocs = choisir_nombre_blocs_convolution(image_size)
    for index_bloc in range(nb_blocs):
        filtres = 32 * (2**index_bloc)
        x = tf.keras.layers.Conv2D(filtres, kernel_size=3, padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(filtres, kernel_size=3, padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
        x = tf.keras.layers.Dropout(0.15 + 0.05 * index_bloc)(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="cnn_chats_chiens")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def preparer_image_prediction(
    path_image: str,
    image_size: Tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
    image = tf.keras.utils.load_img(path_image, target_size=image_size)
    vecteur = tf.keras.utils.img_to_array(image).astype("float32") / 255.0
    return np.expand_dims(vecteur, axis=0)


class ModeleCNNChatsChiens(Modele_classification_chats_chiens):
    """
    CNN binaire entraine from scratch, avec data augmentation artificielle.

    Convention conservee:
      - 0 = chien
      - 1 = chat
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = IMAGE_SIZE,
        batch_size: int = BATCH_SIZE,
        epochs: int = EPOCHS,
        model_path: str = MODEL_PATH_AUGMENTATION,
    ):
        self.image_size = image_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.model_path = model_path
        self.model = construire_cnn(image_size=image_size)

    def entrainer(
        self,
        train_paths: Sequence[str],
        y_train: Sequence[int],
        val_paths: Sequence[str],
        y_val: Sequence[int],
    ) -> tf.keras.callbacks.History:
        train_ds = creer_dataset(train_paths, y_train, self.image_size, self.batch_size, shuffle=True)
        val_ds = creer_dataset(val_paths, y_val, self.image_size, self.batch_size, shuffle=False)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                self.model_path,
                monitor="val_loss",
                save_best_only=True,
            ),
        ]

        effectifs = np.bincount(np.array(y_train, dtype=np.int32), minlength=2)
        nb_total = float(np.sum(effectifs))
        class_weight = {
            LABEL_CHIEN: nb_total / (2.0 * max(effectifs[LABEL_CHIEN], 1)),
            LABEL_CHAT: nb_total / (2.0 * max(effectifs[LABEL_CHAT], 1)),
        }

        return self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.epochs,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1,
        )

    def predire_proba_chat(self, path_image: str) -> float:
        vecteur = preparer_image_prediction(path_image, self.image_size)
        return float(self.model.predict(vecteur, verbose=0)[0][0])

    def predire_label_de_cette_image(self, path_image: str) -> int:
        proba_chat = self.predire_proba_chat(path_image)
        return LABEL_CHAT if proba_chat >= 0.5 else LABEL_CHIEN

    def evaluer(self, val_paths: Sequence[str], y_val: Sequence[int]) -> Dict[str, object]:
        y_proba = np.array([self.predire_proba_chat(path) for path in val_paths])
        y_pred = (y_proba >= 0.5).astype(int)

        return {
            "accuracy": accuracy_score(y_val, y_pred),
            "matrice_confusion": confusion_matrix(y_val, y_pred, labels=[LABEL_CHIEN, LABEL_CHAT]),
            "rapport_classification": classification_report(
                y_val,
                y_pred,
                target_names=[CLASSES[LABEL_CHIEN], CLASSES[LABEL_CHAT]],
                zero_division=0,
            ),
        }


def entrainer_modele_avec_data_augmentation(
    model_path: str = MODEL_PATH_AUGMENTATION,
) -> Tuple[ModeleCNNChatsChiens, Dict[str, object]]:
    train_paths, y_train, val_paths, y_val = charger_data_train_val()
    tf.keras.backend.clear_session()

    modele = ModeleCNNChatsChiens(model_path=model_path)
    modele.entrainer(train_paths, y_train, val_paths, y_val)
    evaluation = modele.evaluer(val_paths, y_val)

    print("\nResultats du modele avec data augmentation")
    print(f"Modele sauvegarde dans: {model_path}")
    print(f"Accuracy validation: {evaluation['accuracy']:.4f}")
    print("Matrice de confusion [[vrai chien, faux chat], [faux chien, vrai chat]]:")
    print(evaluation["matrice_confusion"])
    print(evaluation["rapport_classification"])

    return modele, evaluation


def charger_modele_entraine(
    model_path: str = MODEL_PATH_AUGMENTATION,
) -> ModeleCNNChatsChiens:
    modele = ModeleCNNChatsChiens(model_path=model_path)
    modele.model = tf.keras.models.load_model(model_path)
    return modele


def prediction(
    path_image: str,
    model: Optional[Union[ModeleCNNChatsChiens, tf.keras.Model]] = None,
    model_path: str = MODEL_PATH_AUGMENTATION,
) -> int:
    """Retourne 0 pour chien, 1 pour chat."""
    if model is None:
        model = charger_modele_entraine(model_path)

    if isinstance(model, ModeleCNNChatsChiens):
        return model.predire_label_de_cette_image(path_image)

    vecteur = preparer_image_prediction(path_image)
    proba_chat = float(model.predict(vecteur, verbose=0)[0][0])
    return LABEL_CHAT if proba_chat >= 0.5 else LABEL_CHIEN


def prediction_detaillee(
    path_image: str,
    model: Optional[Union[ModeleCNNChatsChiens, tf.keras.Model]] = None,
    model_path: str = MODEL_PATH_AUGMENTATION,
) -> Dict[str, object]:
    if model is None:
        model = charger_modele_entraine(model_path)

    if isinstance(model, ModeleCNNChatsChiens):
        proba_chat = model.predire_proba_chat(path_image)
    else:
        vecteur = preparer_image_prediction(path_image)
        proba_chat = float(model.predict(vecteur, verbose=0)[0][0])

    label = LABEL_CHAT if proba_chat >= 0.5 else LABEL_CHIEN
    return {
        "label": label,
        "classe": CLASSES[label],
        "proba_chat": proba_chat,
        "proba_chien": 1.0 - proba_chat,
    }


if __name__ == "__main__":
    entrainer_modele_avec_data_augmentation()

