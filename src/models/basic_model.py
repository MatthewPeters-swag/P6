from models.model import Model
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.optimizers import Adam

class BasicModel(Model):
    def _define_model(self, input_shape, categories_count):
        print(">>> BasicModel _define_model running <<<")
        model = Sequential([
            Rescaling(1./255, input_shape=input_shape),

            layers.Conv2D(16, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),

            layers.GlobalAveragePooling2D(),
            layers.Dense(32, activation='relu'),
            layers.Dense(categories_count, activation='softmax')
        ])
        self.model = model

    def _compile_model(self):
        print(">>> BasicModel _compile_model running <<<")
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
