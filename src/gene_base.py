from tensorflow.keras import layers, Input, Model


cm_params = {
    "kernel_size": 4,
    "strides": 2,
    "padding": "same",
    "use_bias": False
}


class EncodeBlock(layers.Layer):
    def __init__(self, n_filter, use_bn=True):
        super(EncodeBlock, self).__init__()
        self.blocks = [
            layers.Conv2D(n_filter, **cm_params),
            layers.LeakyReLU(.2)
        ]
        if use_bn is True:
            self.blocks.insert(1, layers.BatchNormalization())

    def call(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class DecodeBlock(layers.Layer):
    def __init__(self, n_filter, dropout=True, tconv=False):
        super(DecodeBlock, self).__init__()
        self.blocks = [
            layers.BatchNormalization(),
            layers.LeakyReLU(.2)
        ]
        if dropout is True:
            self.blocks.insert(1, layers.Dropout(.3))
        if tconv is True:
            self.blocks.insert(0, layers.Conv2DTranspose(n_filter, **cm_params))
        else:
            self.blocks.insert(0, layers.UpSampling2D(interpolation="bilinear"))
            self.blocks.insert(1, layers.Conv2D(n_filter, 3, 1, "same", use_bias=False))

    def call(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class Encoder(layers.Layer):
    def __init__(self, n_filters=None):
        super(Encoder, self).__init__()
        if n_filters is None:
            n_filters = [64*i for i in [1,2,3,4,4,4,4,4]]

        self.blocks = []
        for i, f in enumerate(n_filters):
            if i == 0:
                self.blocks.append(EncodeBlock(f, use_bn=False))
            else:
                self.blocks.append(EncodeBlock(f))

    def call(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class Decoder(layers.Layer):
    def __init__(self, n_filters=None, out_ch=1):
        super(Decoder, self).__init__()
        if n_filters is None:
            n_filters = [64*i for i in [4,4,4,4,3,2,1]]

        self.blocks = []
        for i, f in enumerate(n_filters):
            if i < 3:
                self.blocks.append(DecodeBlock(f))
            else:
                self.blocks.append(DecodeBlock(f, dropout=False))

        self.blocks.append(layers.UpSampling2D(interpolation="bilinear"))
        self.blocks.append(layers.Conv2D(n_filter, 3, 1, "same", use_bias=False))

    def call(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class EncoderDecoderGenerator(Model):
    def __init__(self, enc_filters=None, dec_filters=None):
        super(EncoderDecoderGenerator, self).__init__()
        self.encoder = Encoder(enc_filters)
        self.decoder = Decoder(dec_filters)

    def call(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def get_summary(self, input_shape=(256,256,1)):
        inputs = Input(input_shape)
        return Model(inputs, self.call(inputs)).summary()


class UNetGenerator(Model):
    def __init__(self. enc_filters=None, dec_filters=None):
        super(UNetGenerator, self).__init__()
        if enc_filters is None:
            enc_filters = [64*i for i in [1,2,3,4,4,4,4,4]]
        if dec_filters is None:
             dec_filters = [64*i for i in [4,4,4,4,3,2,1]]

        self.enc_blocks = []
        for i, f in enumerate(enc_filters):
            if i == 0:
                self.enc_blocks.append(EncodeBlock(f, use_bn=False))
            else:
                self.enc_blocks.append(Encodeblock(f))

        self.dec_blocks = []
        for i, f in enumerate(dec_filters):
            if i < 3:
                self.dec_blocks.append(DecodeBlock(f))
            else:
                self.dec_blocks.append(DecodeBlock(f, dropout=False))

        self.last_upsample = layers.UpSampling2D(interpolation="bilinear")
        self.last_conv = layers.Conv2D(n_filter, 3, 1, "same", use_bias=False)

    def call(self, x):
        features = []
        for block in self.enc_blocks:
            x = block(x)
            features.append(x)

        features = features[:-1]
        for block, feat in zip(self.dec_blocks, features[::-1]):
            x = block(x)
            x = tf.concat([x, feat], axis=-1)

        x = self.last_upsample(x)
        x = self.last_conv(x)
        return x

    def get_summary(self, input_shape=(256,256,1)):
        inputs = Input(input_shape)
        return Model(inputs, self.call(inputs)).summary()
