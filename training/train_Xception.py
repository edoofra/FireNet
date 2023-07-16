import tensorflow as tf
from tensorflow import keras
from utils import parse_configuration
from keras.callbacks import CSVLogger


class Xception():
    def init(self):
        self.config = parse_configuration.parse()
        self.CLASSES = self.config.get("TRAINING", "CLASSES")

    def compile_model(self, model, finetuning=False):

        # define optimizer
        # using legacy opt to avoid slow training on M1 chip
        opt = keras.optimizers.legacy.Adam(lr=0.001, decay=0.001 / 100)
        if finetuning:
            opt = keras.optimizers.legacy.Adam(lr=0.00001, decay=0.00001 / 20)
        recall = tf.keras.metrics.Recall()
        # compile the model
        model.compile(
            optimizer=opt,
            loss=tf.losses.BinaryCrossentropy(from_logits=True),
            metrics=['accuracy', recall])
        return model

    def build_model(self):
        base_model = keras.applications.Xception(weights="imagenet", include_top=False,
                                                 input_shape=(299, 299, 3), pooling="avg")
        base_model.trainable = False
        # add the dense layers at the end of the base model
        final_model = base_model.output
        final_model = keras.layers.Flatten(name="flatten")(final_model)
        final_model = keras.layers.Dropout(0.3)(final_model)
        final_model = keras.layers.BatchNormalization()(final_model)
        final_model = keras.layers.Dense(
            256, activation="relu", kernel_initializer=keras.initializers.HeNormal())(final_model)
        final_model = keras.layers.Dropout(0.3)(final_model)
        final_model = keras.layers.Dense(
            1, activation="sigmoid")(final_model)
        model = keras.Model(inputs=base_model.input, outputs=final_model)
        model = self.compile_model(model)
        return model

    def load_model(self, model_path):
        model = keras.models.load_model(model_path)
        model = self.compile_model(model)
        return model

    def save_model(self, model, model_path):
        model.save(model_path)

    def train(self, model, train_ds, val_ds, NUM_EPOCHS):

        csv_logger = CSVLogger('output/training_log.csv',
                               append=True, separator=';')
        # early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True)

        # train the model
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=NUM_EPOCHS,
            callbacks=[csv_logger, early_stopping]
        )
        return history, model

    def finetuning(self, model, train_ds, val_ds, NUM_EPOCHS):
        # unfreeze all the layer of the base model
        model.trainable = True
        # compile the model
        model = self.compile_model(model, finetuning=True)

        # define the callback to save the best model
        csv_logger = CSVLogger('output/training_finetuning_log.csv',
                               append=True, separator=';')
        # early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True)

        # train the model
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=NUM_EPOCHS,
            callbacks=[csv_logger, early_stopping]
        )
        return history, model

    def evaluate(self, model, test_ds):
        # evaluate the model
        loss, acc, recall = model.evaluate(test_ds)
        return loss, acc, recall
