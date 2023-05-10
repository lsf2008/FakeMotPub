import torch
import torch.nn as nn

from aemodel.base import BaseModule
from aemodel.blocks_3d import DownsampleBlock
from aemodel.blocks_3d import UpsampleBlock
from aemodel.layers.tsc import TemporallySharedFullyConnection
from einops import rearrange
from torchsummary import summary
class Encoder(BaseModule):
    """
    ShanghaiTech model encoder.
    """
    def __init__(self, input_shape, code_length):
        # type: (Tuple[int, int, int, int], int) -> None
        """
        Class constructor:

        :param input_shape: the shape of UCSD Ped2 samples.
        :param code_length: the dimensionality of latent vectors.
        """
        super(Encoder, self).__init__()

        self.input_shape = input_shape
        self.code_length = code_length

        c, t, h, w = input_shape

        activation_fn = nn.LeakyReLU()

        # Convolutional network
        self.conv = nn.Sequential(
            DownsampleBlock(channel_in=c, channel_out=8, activation_fn=activation_fn, stride=(1, 2, 2)),
            DownsampleBlock(channel_in=8, channel_out=16, activation_fn=activation_fn, stride=(1, 2, 1)),
            DownsampleBlock(channel_in=16, channel_out=32, activation_fn=activation_fn, stride=(2, 1, 2)),
            # stride=(1,2,2)-->(2,2,1)
            DownsampleBlock(channel_in=32, channel_out=64, activation_fn=activation_fn, stride=(2, 2, 1)),
            DownsampleBlock(channel_in=64, channel_out=64, activation_fn=activation_fn, stride=(2, 1, 2))
        )

        self.deepest_shape = (64, t // 8, h // 8, w // 8)

    def forward(self, x):
        # types: (torch.Tensor) -> torch.Tensor
        """
        Forward propagation.

        :param x: the input batch of patches.
        :return: the batch of latent vectors.
        """
        # end_out = []
        # h = x
        h = self.conv(x)
        return h

class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, output_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out

class Decoder(BaseModule):
    """
    ShanghaiTech model decoder.
    """
    def __init__(self, code_length, deepest_shape, output_shape):
        # type: (int, Tuple[int, int, int, int], Tuple[int, int, int, int]) -> None
        """
        Class constructor.

        :param code_length: the dimensionality of latent vectors.
        :param deepest_shape: the dimensionality of the encoder's deepest convolutional map.
        :param output_shape: the shape of UCSD Ped2 samples.
        """
        super(Decoder, self).__init__()

        self.code_length = code_length
        self.deepest_shape = deepest_shape
        self.output_shape = output_shape

        dc, dt, dh, dw = deepest_shape
        activation_fn = nn.LeakyReLU()

        # Convolutional network
        self.conv = nn.Sequential(
            UpsampleBlock(channel_in=dc, channel_out=64,
                          activation_fn=activation_fn, stride=(2, 1, 2), output_padding=(1, 0, 1)),
            # stride=(1,2,2)->(2,2,2) padding (0,1,1)_>(1,1,1)
            UpsampleBlock(channel_in=64, channel_out=32,
                          activation_fn=activation_fn, stride=(2, 2, 1), output_padding=(1, 1, 0)),
            UpsampleBlock(channel_in=32, channel_out=16,
                          activation_fn=activation_fn, stride=(2, 1, 2), output_padding=(1, 0, 1)),
            UpsampleBlock(channel_in=16, channel_out=8,
                          activation_fn=activation_fn, stride=(1, 2, 1), output_padding=(0, 1, 0)),
            UpsampleBlock(channel_in=8, channel_out=output_shape[0],
                          activation_fn=activation_fn, stride=(1, 2, 2), output_padding=(0, 1, 1)),
            # nn.Conv3d(in_channels=8, out_channels=output_shape[0], kernel_size=1, bias=False)
        )

    # noinspection LanguageDetectionInspection
    def forward(self, x):
        # types: (torch.Tensor) -> torch.Tensor
        """
        Forward propagation.

        :param x: the batch of latent vectors.
        :return: the batch of reconstructions.
        """
        h = self.conv(x)
        # o = h
        return h
class AeMlp(BaseModule):
    def __init__(self, input_shape, code_length):
        super(AeMlp, self).__init__()
        self.input_shape = input_shape
        self.code_length = code_length

        self.mlp = MLP(self.code_length, 128, 2)

        self.encoder = Encoder(
            input_shape=input_shape,
            code_length=code_length
        )

        # Build decoder
        self.decoder = Decoder(
            code_length=code_length,
            deepest_shape=self.encoder.deepest_shape,
            output_shape=input_shape
        )

    def forward(self, x):
        '''
        :param x: b, c, t, h, w
        :return: out_mlp: (b*t*h*w) * 2; out_enc: (b, c, t, h, w)
        '''
        # b, c,t, h, w = x.shape
        out_enc = self.encoder(x)

        out_mlp = rearrange(out_enc, 'b c t h w->b t h w c')
        out_mlp = self.mlp(out_mlp)
        # out = torch.einsum('bthwc->bcthw', out)

        out_mlp = rearrange(out_mlp, 'b t h w c->(b t h w) c')
        # softmax for crossentropy loss
        out_mlp = torch.nn.functional.softmax(out_mlp, dim =1)

        # out_decoder = rearrange(out, 'b t h w c->b c t h w')

        out_enc = self.decoder(out_enc)
        # divide into 2 branches
        # out = out.reshape((b, ))
        return out_mlp, out_enc

if __name__ =='__main__':
    torch.backends.cudnn.enabled = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.rand((2, 3, 8, 48, 48)).to(device)
    input_shape =[3, 8, 48, 48]
    code_length = 64
    model = AeMlp(input_shape, code_length).to(device)
    soft, dec = model(x)
    # summary(model, input_shape)
    print(soft.shape, dec.shape)