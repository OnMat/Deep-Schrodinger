
import tensorflow as tf
dtype = tf.float64

class LSTMLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, input_dim, trans1="tanh", trans2="tanh",**kwargs):
        """
        Args:
            input_dim (int):       dimensionalidade da entrada
            output_dim (int):      número de saídas da LSTM
            trans1, trans2 (str):  funções de ativação internas ("tanh", "relu", "sigmoid")
        """
        super(LSTMLayer, self).__init__(dtype=dtype,**kwargs)

        self.output_dim = output_dim
        self.input_dim = input_dim
        self.trans1_name = trans1
        self.trans2_name = trans2

        self.output_dim = output_dim
        self.input_dim = input_dim

        # ativação
        self.trans1 = self._get_activation(trans1)
        self.trans2 = self._get_activation(trans2)

        # Pesos para entrada X
        self.Uz = self.add_weight(name="Uz", shape=(self.input_dim, self.output_dim),
                                  initializer=tf.keras.initializers.GlorotUniform(), trainable=True)
        self.Ug = self.add_weight(name="Ug", shape=(self.input_dim, self.output_dim),
                                  initializer=tf.keras.initializers.GlorotUniform(), trainable=True)
        self.Ur = self.add_weight(name="Ur", shape=(self.input_dim, self.output_dim),
                                  initializer=tf.keras.initializers.GlorotUniform(), trainable=True)
        self.Uh = self.add_weight(name="Uh", shape=(self.input_dim, self.output_dim),
                                  initializer=tf.keras.initializers.GlorotUniform(), trainable=True)

        # Pesos para saída anterior S
        self.Wz = self.add_weight(name="Wz", shape=(self.output_dim, self.output_dim),
                                  initializer=tf.keras.initializers.GlorotUniform(), trainable=True)
        self.Wg = self.add_weight(name="Wg", shape=(self.output_dim, self.output_dim),
                                  initializer=tf.keras.initializers.GlorotUniform(), trainable=True)
        self.Wr = self.add_weight(name="Wr", shape=(self.output_dim, self.output_dim),
                                  initializer=tf.keras.initializers.GlorotUniform(), trainable=True)
        self.Wh = self.add_weight(name="Wh", shape=(self.output_dim, self.output_dim),
                                  initializer=tf.keras.initializers.GlorotUniform(), trainable=True)

        # Bias
        self.bz = self.add_weight(name="bz", shape=(1, self.output_dim),
                                  initializer=tf.keras.initializers.Zeros(), trainable=True)
        self.bg = self.add_weight(name="bg", shape=(1, self.output_dim),
                                  initializer=tf.keras.initializers.Zeros(), trainable=True)
        self.br = self.add_weight(name="br", shape=(1, self.output_dim),
                                  initializer=tf.keras.initializers.Zeros(), trainable=True)
        self.bh = self.add_weight(name="bh", shape=(1, self.output_dim),
                                  initializer=tf.keras.initializers.Zeros(), trainable=True)

    def _get_activation(self, name):
        """Retorna a função de ativação do TF correspondente ao nome"""
        if name == "tanh":
            return tf.nn.tanh
        elif name == "relu":
            return tf.nn.relu
        elif name == "sigmoid":
            return tf.nn.sigmoid
        else:
            return None

    def call(self, S, X):
        """
        Args:
            S: saída da camada anterior
            X: entrada atual
        """
        Z = self.trans1(tf.matmul(X, self.Uz) + tf.matmul(S, self.Wz) + self.bz)
        G = self.trans1(tf.matmul(X, self.Ug) + tf.matmul(S, self.Wg) + self.bg)
        R = self.trans1(tf.matmul(X, self.Ur) + tf.matmul(S, self.Wr) + self.br)
        H = self.trans2(tf.matmul(X, self.Uh) + tf.matmul(tf.multiply(S, R), self.Wh) + self.bh)

        S_new = tf.multiply(1.0 - G, H) + tf.multiply(Z, S)
        return S_new

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "input_dim": self.input_dim,
            "trans1": "tanh",  
            "trans2": "tanh"
        })
        return config

    @classmethod
    def from_config(cls, config):
        config.pop("dtype", None)   
        return cls(**config)


class DenseLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, input_dim, transformation=None,**kwargs):
        """
        Args:
            input_dim:       dimensionalidade de entrada
            output_dim:      número de saídas da camada densa
            transformation:  função de ativação ("tanh", "relu" ou None)
        """
        super(DenseLayer, self).__init__(dtype = tf.float64,**kwargs )

        self.output_dim = output_dim
        self.input_dim = input_dim
        self.transformation_name = transformation

        self.output_dim = output_dim
        self.input_dim = input_dim

        # Pesos com inicialização Xavier (GlorotUniform)
        self.W = self.add_weight(
            name="W",
            shape=(self.input_dim, self.output_dim),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True
        )

        # Bias
        self.b = self.add_weight(
            name="b",
            shape=(1, self.output_dim),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True
        )

        # Função de ativação
        if transformation == "tanh":
            self.transformation = tf.nn.tanh
        elif transformation == "relu":
            self.transformation = tf.nn.relu
        else:
            self.transformation = None

    def call(self, X):
        """
        Computa a saída da camada para uma entrada X
        """
        S = tf.matmul(X, self.W) + self.b
        if self.transformation:
            S = self.transformation(S)
        return S

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "input_dim": self.input_dim,
            "transformation": (
                "tanh" if self.transformation == tf.nn.tanh
                else "relu" if self.transformation == tf.nn.relu
                else None
            )
        })
        return config

    @classmethod
    def from_config(cls, config):
        config.pop("dtype", None)   
        return cls(**config)



class DGMNet1D(tf.keras.Model):
    def __init__(self, layer_width, n_layers, input_dim, final_trans=None,**kwargs):
        """
        Args:
            layer_width: largura das camadas intermediárias
            n_layers: número de camadas LSTM intermediárias
            input_dim: dimensão espacial da entrada (EXCLUI tempo)
            final_trans: função de ativação da camada final
        """
        super(DGMNet1D, self).__init__(dtype = tf.float64,**kwargs)

        self.layer_width = layer_width
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.final_trans = final_trans

        # camada inicial totalmente conectada (Dense)
        # geralmente adicionamos +1 para entrada de tempo mas não é o caso
        self.initial_layer = DenseLayer(layer_width, input_dim, transformation="tanh")

        # camadas LSTM intermediárias
        self.LSTMLayerList = [
            LSTMLayer(layer_width, input_dim) for _ in range(n_layers)
        ]

        # camada final totalmente conectada com saída única
        self.final_layer = DenseLayer(1, layer_width, transformation=final_trans)

    def call(self, x):
        """
        Propagação direta pelo DGMNet
        Args:
            X: entrada (tensor de dimensão [batch_size, input_dim+1])
        Returns:
            Saída da rede
        """

        X = x 
        # camada inicial
        S = self.initial_layer(X)

        # camadas LSTM intermediárias
        for lstm_layer in self.LSTMLayerList:
            S = lstm_layer(S, X)

        # camada final
        output = self.final_layer(S)

        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "layer_width": self.layer_width,
            "n_layers": self.n_layers,
            "input_dim": self.input_dim,
            "final_trans": self.final_trans
        })
        return config

    @classmethod
    def from_config(cls, config):
        config.pop("dtype", None)
        return cls(**config)