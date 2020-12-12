from tensorflow.keras import layers, Input, Model


class DiscBlock(layers.Layer):
    def __init__(
        self, n_filter, strides=2, custom_pad=False, use_bn=False, activation=True
        ):
        super(DiscBlock, self).__init__()
        self.blocks = []
        if custom_pad is True:
            self.blocks.append(layers.ZeroPadding2D())
            self.blocks.append(layers.Conv2D(n_filter, 4, strides, "valid", use_bias=False))
        else:
            self.blocks.append(layers.Conv2D(n_filter, 4, strides, "same", use_bias=False))
        if use_bn is True:
            self.blocks.append(layers.BatchNormalization())
        if activation is True:
            self.blocks.append(layers.LeakyReLU(.2))

    def call(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class UNetDiscriminator(Model):
    def __init__(self, n_filters=None):
        super(Discriminator, self).__init__()
        if n_filters is None:
            n_filters = [64, 128, 256, 512, 1]

        self.concat = layers.Concatenate()
        self.blocks = []

        for i, f in enumerate(n_filters):
            self.blocks.append(
                DiscBlock(
                    n_filter=f,
                    strides=2 if i<3 else 1,
                    custom_pad=False if i<3 else True,
                    use_bn=False if i==0 and i==4 else True,
                    activation=True if i<4 else False
                )
            )
        self.sigmoid = layers.Activation("sigmoid")

    def call(self, x, y):
        out = self.concat([x, y])
        for block in self.blocks:
            out = block(out)
        return self.sigmoid(out)

    def get_summary(self, x_shape=(256,256,1)):
        x, y = Input(x_shape), Input(x_shape)
        return Model((x, y), self.call(x, y)).summary()


class ResNetDiscriminator(Model):
    def __init__(self, n_filter=64):
        super(ResNetDiscriminator, self).__init__()
        # C64, C128, C256, C512
        self.blocks = Sequential()
        for _ in range(4):
            self.blocks.add(layers.Conv2D(n_filter, 4, 2, "same", use_bias=False))
            if n_filter != 64:
                self.blocks.add(InstanceNormalization())
            self.blocks.add(layers.LeakyReLU(.2))
            n_filter *= 2
        self.conv = layers.Conv2D(1, 4, 1, "same")

    def call(self, x):
        out = self.blocks(x)
        return self.conv(out)
