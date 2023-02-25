import torch
import torch.nn as nn



#https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial9/AE_CIFAR10.html
class Encoder(nn.Module):

    def __init__(self,
                 num_input_channels : int=4,
                 base_channel_size : int=8,
                 latent_dim : int =32,
                 act_fn : nn.Module = nn.GELU):
        """
        Inputs:
            - num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, 16, kernel_size=3, padding=1, stride=2), # 256x256 => 128x128
            act_fn(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2), #128x128 => 64x64
            act_fn(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2), #64x64 => 32x32
            act_fn(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2), #32x32 => 16 x 16
            act_fn(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2), # 16x16 => 8x8
            act_fn(),
            # nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            # act_fn(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2), # 8x8 => 4x4
            act_fn(),
            nn.Flatten(), # Image grid to single feature vector
            nn.Linear(16*512, latent_dim)
        )

    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):

    def __init__(self,
                 num_input_channels : int = 4,
                 base_channel_size : int = 8,
                 latent_dim : int = 32,
                 act_fn : nn.Module = nn.GELU):
        """
        Inputs:
            - num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 16*512),
            act_fn()
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, output_padding=1, padding=1, stride=2), # 4x4 => 8x8
            act_fn(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, output_padding=1, padding=1, stride=2), # 8x8 => 16x16
            act_fn(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d( 128, 64, kernel_size=3, output_padding=1, padding=1, stride=2),  # 16x16 => 32x32
            act_fn(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, output_padding=1, padding=1, stride=2),  # 32x32 => 64x64
            act_fn(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, output_padding=1, padding=1, stride=2),  # 64x64 => 128x128
            act_fn(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(16, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2), # 128x128 => 256x256
            nn.Sigmoid() # The input images is scaled between 0 and 1, hence the output has to be bounded as well
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 4, 4)
        x = self.net(x)
        return x



# encoder=Encoder().to('cpu')
# decoder=Decoder().to('cpu')
# x=torch.randn((5,4,256,256))
# z=torch.randn(5,32)
#
# x_hat=decoder(z)

