import torch
import torch.nn as nn

from aemodel.base import BaseModule
from aemodel.blocks_3d import DownsampleBlock
from aemodel.blocks_3d import UpsampleBlock
from aemodel.layers.tsc import TemporallySharedFullyConnection
from einops import rearrange
# from torchsummary import summary
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

class MLP2out(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP2out, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        # self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc31 = torch.nn.Linear(hidden_dim, 2)
        self.fc3 = torch.nn.Linear(hidden_dim, output_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        '''
        Parameters
        ----------
        x
        Returns
        -------
        out1:   for cross entropy
        out2:   for decoder
        '''
        out = self.relu(self.fc1(x))
        # out = self.relu(self.fc2(out))
        out1 = self.fc31(out)
        out2 = self.fc3(out)
        return out1, out2
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
        '''
        x -->Encoder -->MLP--->x'
                     -->Decoder --->\hat{x}
        Parameters
        ----------
        input_shape
        code_length
        '''
        super(AeMlp, self).__init__()
        self.input_shape = input_shape
        self.code_length = code_length

        self.mlp = MLP(self.code_length, 32, 2)

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
class Ae2Mlps(BaseModule):
    def __init__(self, input_shape, code_length):
        '''
        x -->Encoder -->MLP--->x'
        x -->Encoder -->MLP--->x1 --->Decoder --->\hat{x}
        Parameters
        ----------
        input_shape
        code_length
        '''
        super(Ae2Mlps, self).__init__()
        self.input_shape = input_shape
        self.code_length = code_length

        # x-->encoder -->MLP -->cross entropy
        self.mlp_cross = MLP(self.code_length, 32, 2)
        # x-->encoder -->MLP -->decoder
        self.mlp_decoder = MLP(self.code_length, 128, 64)

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
        out_mlp_cross = self.mlp_cross(out_mlp)
        # out = torch.einsum('bthwc->bcthw', out)

        out_mlp_cross = rearrange(out_mlp_cross, 'b t h w c->(b t h w) c')
        # softmax for crossentropy loss
        # out_mlp_cross = torch.nn.functional.softmax(out_mlp_cross, dim =1)

        # out_decoder = rearrange(out, 'b t h w c->b c t h w')
        out_decoder = self.mlp_decoder(out_mlp)
        out_decoder = rearrange(out_decoder, 'b t h w c->b c t h w')
        out_decoder = self.decoder(out_decoder)

        # divide into 2 branches
        # out = out.reshape((b, ))
        return out_mlp_cross, out_decoder

class Ae1Mlp2(BaseModule):
    def __init__(self, input_shape, code_length, mlp_hidden=32):
        '''
        x -->Encoder -->MLP--->x1'
                            -->x2 --->Decoder --->\hat{x}
        Parameters
        ----------
        input_shape
        code_length
        '''
        super(Ae1Mlp2, self).__init__()
        self.input_shape = input_shape
        self.code_length = code_length

        self.mlp = MLP2out(self.code_length, mlp_hidden, self.code_length)

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
        out_mlp, out_dec = self.mlp(out_mlp)
        # out = torch.einsum('bthwc->bcthw', out)

        out_mlp = rearrange(out_mlp, 'b t h w c->(b t h w) c')
        # softmax for crossentropy loss
        # out_mlp = torch.nn.functional.softmax(out_mlp, dim =1)

        out_dec = rearrange(out_dec, 'b t h w c->b c t h w')

        out_dec = self.decoder(out_dec)

        return out_mlp, out_dec

# ==================AE+motion==================
class MLP3out(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP3out, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)

        # for motion classfication
        self.fc_mot_cls = torch.nn.Linear(hidden_dim, 2)
        # for motion
        self.fc_mot_rec = torch.nn.Linear(hidden_dim, output_dim)
        # for appearance
        self.fc_app = torch.nn.Linear(hidden_dim, output_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        '''
        Parameters
        ----------
        x
        Returns
        -------
        out1:   for cross entropy
        out2:   for decoder
        '''
        out = self.relu(self.fc1(x))
        # out = self.relu(self.fc2(out))
        out_mot_cls = self.fc_mot_cls(out)
        out_mot_rec = self.fc_mot_rec(out)
        out_app = self.fc_app(out)
        return out_mot_cls, out_mot_rec, out_app

class Ae1Mlp3(BaseModule):
    def __init__(self, input_shape, code_length):
        '''
        x -->Encoder -->MLP--->x1'
                            -->x2 --->Decoder --->\hat{x}
        Parameters
        ----------
        input_shape
        code_length
        '''
        super(Ae1Mlp3, self).__init__()
        self.input_shape = input_shape
        self.code_length = code_length

        self.mlp = MLP3out(self.code_length, 128, 64)

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
        out_mlp, out_mot_rec, out_app = self.mlp(out_mlp)
        # out = torch.einsum('bthwc->bcthw', out)

        out_mlp = rearrange(out_mlp, 'b t h w c->(b t h w) c')
        # softmax for crossentropy loss
        # out_mlp = torch.nn.functional.softmax(out_mlp, dim =1)

        out_mot_rec = rearrange(out_mot_rec, 'b t h w c->b c t h w')
        out_app = rearrange(out_app, 'b t h w c->b c t h w')
        out_join = torch.cat((out_mot_rec, out_app), dim=0)


        out_join = self.decoder(out_join)

        out_mot_rec, out_app = torch.split(out_join, out_app.shape[0], dim=0)
        # out_app = self.decoder(out_app)

        return out_mlp, out_mot_rec, out_app

class Ae3Mlps(BaseModule):
    def __init__(self, input_shape, code_length):
        '''
        x -->Encoder -->MLP--->x'
        x -->Encoder -->MLP--->x1 --->Decoder --->\hat{x}
        Parameters
        ----------
        input_shape
        code_length
        '''
        super(Ae3Mlps, self).__init__()
        self.input_shape = input_shape
        self.code_length = code_length

        # x-->encoder -->MLP -->cross entropy
        self.mlp_cross = MLP(self.code_length, 32, 2)
        # x-->encoder -->MLP -->decoder
        self.mlp_mot_decoder = MLP(self.code_length, 128, 64)

        # x-->encoder -->MLP -->decoder
        self.mlp_app_decoder = MLP(self.code_length, 128, 64)

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
        out_mlp_cross = self.mlp_cross(out_mlp)
        # out = torch.einsum('bthwc->bcthw', out)
        out_mlp_cross = rearrange(out_mlp_cross, 'b t h w c->(b t h w) c')

        # out_decoder = rearrange(out, 'b t h w c->b c t h w')
        out_decoder = self.mlp_mot_decoder(out_mlp)
        out_decoder = rearrange(out_decoder, 'b t h w c->b c t h w')
        out_decoder = self.decoder(out_decoder)

        out_app_decoder = self.mlp_app_decoder(out_mlp)
        out_app_decoder = rearrange(out_app_decoder, 'b t h w c->b c t h w')
        out_app_decoder = self.decoder(out_app_decoder)

        # divide into 2 branches
        # out = out.reshape((b, ))
        return out_mlp_cross, out_decoder, out_app_decoder

class Ae0Mlp(BaseModule):
    def __init__(self, input_shape, code_length, mlp_hidden=32):
        '''
        x -->Encoder -->MLP--->x1'
                            -->x2 --->Decoder --->\hat{x}
        Parameters
        ----------
        input_shape
        code_length
        '''
        super(Ae0Mlp, self).__init__()
        self.input_shape = input_shape
        self.code_length = code_length

        self.fc = torch.nn.Linear(self.code_length, 2)
            # MLP2out(self.code_length, mlp_hidden, self.code_length)

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
        out_mlp, out_dec = self.fc(out_mlp)
        # out = torch.einsum('bthwc->bcthw', out)

        out_mlp = rearrange(out_mlp, 'b t h w c->(b t h w) c')
        # softmax for crossentropy loss
        out_mlp = self.fc(out_mlp)

        out_dec = rearrange(out_dec, 'b t h w c->b c t h w')

        out_dec = self.decoder(out_dec)

        return out_mlp, out_dec

if __name__ =='__main__':
    torch.backends.cudnn.enabled = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.rand((2, 3, 8, 48, 48)).to(device)
    input_shape =[3, 8, 48, 48]
    code_length = 64
    # model = AeMlp(input_shape, code_length).to(device)
    model = Ae1Mlp2(input_shape, code_length).to(device)
    soft, dec = model(x)
    # summary(model, input_shape)
    print(soft.shape, dec.shape)