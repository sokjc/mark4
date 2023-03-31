from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Model, Sequential

class MyModel(Model): 

    def __init__(self, num_layers, num_nodes, num_classes) -> None:
        super(MyModel, self).__init__()
        self.num_layers = num_layers
        self.nodes_per_layer = num_nodes
        self.classes = num_classes
        self.hidden = Sequential([Dense(self.nodes_per_layer, activation='relu') for i in enumerate(self.layers)])
        self.logits = Dense(self.classes)

    def call(self, inputs):
        x = Flatten(input_shape=(28,28))(inputs)
        x = self.hidden(x)
        x = self.logits(x)
        return x