from tensorflow.keras import layers, Input, Model, Sequential
from src.layers import (
    ReflectionPadding2D,
    InstanceNormalization
)


s1_params = {
    "kernel_size": 3,
    "strides": 1,
    "padding": "valid",
    "use_bias": False
}
s2_params = s1_params.copy()
s2_params["strides"] = 2


class ResidualEncodeBlock(layers.Layer):
    def __init__(self, n_filter):
        super(ResidualEncodeBlock, self).__init__()
        self.blocks = Sequential()
        for i in range(2):
            self.blocks.add(ReflectionPadding2D(1))
            self.blocks.add(layers.Conv2D(n_filter, **s1_params))
            self.blocks.add(InstanceNormalization())
            if i == 0:
                self.blocks.add(layers.ReLU())
        self.add = layers.Add()

    def call(self, x):
        out = x
        out = self.blocks(out)
        return self.add([x, out])


class ResNetGenerator(Model):
    def __init__(self, n_filter=64, n_downs=2, n_resblocks=9, tconv=False):
        super(ResNetGenerator, self).__init__()
        # c7s1-64
        self.blocks0 = Sequential([
            ReflectionPadding2D(3),
            layers.Conv2D(n_filter, 7, 1, "valid", use_bias=False),
            InstanceNormalization(),
            layers.ReLU()
        ])
        # d128, d256
        self.blocks1 = Sequential()
        for _ in range(n_downs):
            n_filter *= 2
            self.blocks1.add(layers.Conv2D(n_filter, **s2_params))
            self.blocks1.add(InstanceNormalization())
            self.blocks1.add(layers.ReLU())
        # R256 * 9
        self.blocks2 = Sequential()
        for _ in range(n_resblocks):
            self.blocks2.add(ResidualEncodeBlock(n_filter))
        # u128, u64
        self.blocks3 = Sequential()
        for _ in range(n_downs):
            n_filter //= 2
            if tconv:
                self.blocks3.add(layers.Conv2DTranspose(n_filter,**s2_params))
            else:
                self.blocks3.add(layers.UpSampling2D(interpolation="bilinear"))
                self.blocks3.add(layers.Conv2D(n_filter, 3, 1, "same", use_bias=False))
            self.blocks3.add(InstanceNormalization())
            self.blocks3.add(layers.ReLU())
        # c7s1-3
        self.blocks4 = Sequential([
            ReflectionPadding2D(3),
            layers.Conv2D(1, 7, 1, "valid", use_bias=False),
            InstanceNormalization(),
            layers.ReLU()
        ])

    def call(self, x):
        out = self.blocks0(x)
        out = self.blocks1(out)
        out = self.blocks2(out)
        out = self.blocks3(out)
        return self.blocks4(out)
