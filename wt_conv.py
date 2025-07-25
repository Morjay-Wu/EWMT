import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import pywt
import pywt.data



def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    # Get the specified wavelet type
    w = pywt.Wavelet(wave)
    
    # Create decomposition high-pass and low-pass filters
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)  # High-pass filter
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)  # Low-pass filter
    
    # Construct four 2D filters (LL, LH, HL, HH) for 2D Wavelet Transform
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),   # LL: low-low
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),   # LH: low-high
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),   # HL: high-low
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)],  # HH: high-high
                              dim=0)
    
    # Repeat filters for all input channels
    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    # Create reconstruction high-pass and low-pass filters
    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])  # High-pass filter
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])  # Low-pass filter
    
    # Construct four filters for 2D Inverse Wavelet Transform
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),   # LL
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),   # LH
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),   # HL
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)],  # HH
                              dim=0)
    
    # Repeat filters for all output channels
    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters

def wavelet_transform(x, filters):

    b, c, h, w = x.shape
    # Calculate padding for the filters
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    # Apply convolution for wavelet transform, using stride=2 for downsampling
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    # Reshape the tensor to separate the four sub-bands for further processing
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x

def inverse_wavelet_transform(x, filters):

    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    # Reshape the tensor, merging channel and sub-band dimensions
    x = x.reshape(b, c * 4, h_half, w_half)
    # Use transposed convolution for upsampling and reconstruction
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x

class WTConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):

        super(WTConv2d, self).__init__()

        # Current implementation requires input and output channels to be the same.
        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        # Create wavelet and inverse wavelet filters
        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        # Set them as non-trainable parameters
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        # Define wavelet and inverse wavelet transform functions
        self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)

        # Base convolution layer (applied to the original input)
        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1,
                                   groups=in_channels, bias=bias)
        # Scaling factor for the base convolution path
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        # Convolution layers applied at each wavelet level
        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1, dilation=1,
                       groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)]
        )
        # Scaling factors for each wavelet level, initialized to a small value for stable training
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        # Handle stride > 1
        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride,
                                                   groups=in_channels)
        else:
            self.do_stride = None

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor [B, C, H, W].
            
        Returns:
            torch.Tensor: Processed tensor [B, C, H/stride, W/stride].
        """
        # Store low and high-frequency components at each level
        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        # The low-frequency component for the current level, initialized with the input x
        curr_x_ll = x

        # Top-down: perform wavelet decomposition level by level
        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            
            # Pad if the dimensions are odd
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            # Wavelet transform
            curr_x = self.wt_function(curr_x_ll)
            # Extract the low-frequency component (LL)
            curr_x_ll = curr_x[:, :, 0, :, :]

            # Apply convolution to all components
            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            # Store the processed low and high-frequency components
            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])  # LL
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :]) # LH, HL, HH

        # Bottom-up: perform inverse wavelet reconstruction level by level
        next_x_ll = 0  # Initialize the low-frequency component for the next level up

        for i in range(self.wt_levels - 1, -1, -1):
            # Get components and shape for the current level
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            # Add the low-frequency component from the level below
            curr_x_ll = curr_x_ll + next_x_ll

            # Reconstruct the current level
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)

            # Crop to the original size
            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        # The result from the wavelet processing path
        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        # Process through the original convolution path
        x = self.base_scale(self.base_conv(x))
        
        # Combine the results from both paths
        x = x + x_tag

        # Apply stride if necessary
        if self.do_stride is not None:
            x = self.do_stride(x)

        return x

class _ScaleModule(nn.Module):
    """
    A scaling module to weight features.
    """
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        """
        Initializes the scaling module.
        
        Args:
            dims (tuple): Dimensions of the scaling parameter.
            init_scale (float): Initial value for the scaling parameter.
            init_bias: Not used.
        """
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        """
        Scales the input.
        """
        return torch.mul(self.weight, x)

# Test code: Input N C H W, Output N C H W
if __name__ == '__main__':
    # Create an instance of WTConv2d
    block = WTConv2d(32, 32)
    # Create a random input tensor
    input_tensor = torch.rand(1, 32, 64, 64)
    # Forward pass
    output = block(input_tensor)
    # Print shapes
    print("input.shape:", input_tensor.shape)
    print("output.shape:", output.shape)