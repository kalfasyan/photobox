import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from natsort import index_natsorted
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import (Activation, Dense, Dropout,
                                     GlobalAveragePooling2D)
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

from utils import (copy_files, non_max_suppression_fast, overlay_yolo,
                   save_insect_crops)

logger = logging.getLogger(__name__)


class ModelDetections(object):

    def __init__(self, path, img_dim=150, target_classes=['bl', 'c', 'k', 'm', 'sw', 't', 'v', 'wmv']):
        self.path = str(path)
        self.df = pd.DataFrame({"filepath": list(Path(self.path).rglob('*.png'))}).astype(str)
        self.df['insect_id'] = self.df.filepath.apply(lambda x: x.split('/')[-1].split('_')[-1][:-4])
        self.target_classes = target_classes
        self.le = LabelEncoder()
        self.le.fit(self.target_classes)
        self.nb_classes = len(self.target_classes)
        self.img_dim = img_dim

    def create_data_generator(self):
        datagen = ImageDataGenerator(rescale=1./255, fill_mode="nearest")
        generator = datagen.flow_from_dataframe(self.df, 
                                                x_col='filepath', 
                                                y_col='insect_id',
                                                target_size=(self.img_dim, self.img_dim), 
                                                batch_size=1, 
                                                class_mode='categorical', 
                                                shuffle=False,
                                                random_state=42)
        self.generator = generator

    def get_predictions(self, model, stickyplatehandler, certainty_threshold=90):
        df = self.df
        assert hasattr(self, 'generator'), 'Create data generator first.'

        pred = model.predict(self.generator)
        y_pred = np.argmax(pred, axis=1)

        df['prediction'] = self.le.inverse_transform(y_pred)
        df = pd.concat([df,pd.DataFrame(pred*100, columns=[i for i in self.target_classes])], axis=1)
        df = df.merge(stickyplatehandler.yolo_specs, on='insect_id')
        df['uncertain'] = df[self.target_classes].max(axis=1) < certainty_threshold
        df['class'] = 'na'
        df['top_prob'] = df[self.target_classes].max(axis=1)


        self.df = df.sort_values(
            by="insect_id",
            key=lambda x: np.argsort(index_natsorted(x))
        ).reset_index(drop=True)


class InsectModel(object):

    def __init__(self, path, img_dim, nb_classes):
        self.path = path
        self.img_dim = img_dim
        self.nb_classes = nb_classes

    def load(self):
        base_model = EfficientNetB0(include_top=False, weights='imagenet', 
                                input_shape=(self.img_dim,self.img_dim,3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)
        x = Dense(self.nb_classes, activation='relu')(x)
        predictions = Activation('softmax')(x)        

        model = Model(inputs=base_model.input, outputs=predictions)

        model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['Accuracy'])
        model.load_weights(self.path)

        return model

    def predict(self, generator, batch_size=1, verbose=1):
        return self.model.predict(generator, batch_size=batch_size, verbose=verbose)
