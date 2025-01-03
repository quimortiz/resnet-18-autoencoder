import torch.nn as nn
import torch

from classes.resnet_using_basic_block_encoder import Encoder, BasicBlockEnc
from classes.resnet_using_basic_block_decoder import Decoder, BasicBlockDec
from classes.resnet_using_light_basic_block_encoder import (
    LightEncoder,
    LightBasicBlockEnc,
)
from classes.resnet_using_light_basic_block_decoder import (
    LightDecoder,
    LightBasicBlockDec,
)


class AE(nn.Module):
    """Construction of resnet autoencoder.

    Attributes:
        network (str): the architectural type of the network. There are 2 choices:
            - 'default' (default), related with the original resnet-18 architecture
            - 'light', a samller network implementation of resnet-18 for smaller input images.
        num_layers (int): the number of layers to be created. Implemented for 18 layers (default) for both types
            of network, 34 layers for default only network and 20 layers for light network.
    """

    def __init__(self, network="default", nz=8, num_layers=18, weights=None):
        """Initialize the autoencoder.

        Args:
            network (str): a flag to efine the network version. Choices ['default' (default), 'light'].
             num_layers (int): the number of layers to be created. Choices [18 (default), 34 (only for
                'default' network), 20 (only for 'light' network).
        """
        super().__init__()
        self.network = network
        self.weights = weights
        if self.network == "default":
            if num_layers == 18:
                # resnet 18 encoder
                self.encoder = Encoder(BasicBlockEnc, [2, 2, 2, 2])
                # resnet 18 decoder
                self.decoder = Decoder(BasicBlockDec, [2, 2, 2, 2])
            elif num_layers == 34:
                # resnet 34 encoder
                self.encoder = Encoder(BasicBlockEnc, [3, 4, 6, 3])
                # resnet 34 decoder
                self.decoder = Decoder(BasicBlockDec, [3, 4, 6, 3])
            else:
                raise NotImplementedError(
                    "Only resnet 18 & 34 autoencoder have been implemented for images size >= 64x64."
                )
        elif self.network == "light":
            if num_layers == 18:
                # resnet 18 encoder
                self.encoder = LightEncoder(LightBasicBlockEnc, [2, 2, 2])
                # resnet 18 decoder
                self.decoder = LightDecoder(LightBasicBlockDec, [2, 2, 2])
            elif num_layers == 20:
                # resnet 18 encoder
                self.encoder = LightEncoder(LightBasicBlockEnc, [3, 3, 3])
                # resnet 18 decoder
                self.decoder = LightDecoder(LightBasicBlockDec, [3, 3, 3])
            else:
                raise NotImplementedError(
                    "Only resnet 18 & 20 autoencoder have been implemented for images size < 64x64."
                )
        else:
            raise NotImplementedError(
                "Only default and light resnet have been implemented. Th light version corresponds to input datasets with size less than 64x64."
            )

        # Convolution layer to reduce channels from 64 to 32
        self.enc_conv = nn.Conv2d(
            in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1
        )

        # Pooling layer to reduce spatial dimensions from 16x16 to 8x8
        self.enc_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layer to reduce from 32 * 8 * 8 = 2048 to 8 dimensions
        self.enc_fc = nn.Linear(32 * 8 * 8, nz)

        # Fully connected layer to expand 8-dim vector to 32 * 8 * 8
        self.dec_fc = nn.Linear(nz, 32 * 8 * 8)

        # self.deconv = nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)

        # Transposed conv layer to go from 32x8x8 to 64x16x16
        self.dec_deconv1 = nn.ConvTranspose2d(
            in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1
        )
        self.dec_bn1 = nn.BatchNorm2d(64)

        # Transposed conv layer to go from 64x16x16 to 64x32x32
        self.dec_deconv2 = nn.ConvTranspose2d(
            in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1
        )
        self.dec_bn2 = nn.BatchNorm2d(64)

        if self.weights is not None:
            self.load_state_dict(torch.load(self.weights)["vision_state_dict"])
            self.eval()
            print("weights loaded!")

    def encode(self, x):
        """ """
        z = self.encoder(x)

        # print("original z" , z.shape)
        #
        z = self.enc_conv(z)
        z = self.enc_pool(z)
        z = z.view(-1, 32 * 8 * 8)
        z = self.enc_fc(z)
        return z, None, None

    def decode(self, z):
        """ """

        y = self.dec_fc(z)
        y = y.view(y.size(0), 32, 8, 8)  # Reshape to (batch_size, 32, 8, 8)

        # First transposed convolution: (batch_size, 32, 8, 8) -> (batch_size, 64, 16, 16)
        y = self.dec_deconv1(y)
        y = self.dec_bn1(y)
        y = nn.ReLU()(y)

        # Second transposed convolution: (batch_size, 64, 16, 16) -> (batch_size, 64, 32, 32)
        # y = self.dec_deconv2(y)
        # y = self.dec_bn2(y)
        # y = nn.ReLU()(y)
        # print("yshape" , y.shape)
        y = self.decoder(y)
        return y

    def forward(self, x):
        """The forward functon of the model.

        Args:
            x (torch.tensor): the batched input data

        Returns:
            x (torch.tensor): encoder result
            z (torch.tensor): decoder result
        """
        z, _, _ = self.encode(x)

        y = self.decode(z)

        return y, z
