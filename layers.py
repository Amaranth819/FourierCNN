import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

# '''
#     Fourier convolutional layer 1.0
# '''
# class FourierConv2D(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, bias = True):
#         super(FourierConv2D, self).__init__()

#         self.real_kernel = nn.Parameter(data = torch.Tensor(out_channels, in_channels, kernel_size, kernel_size), requires_grad = True)
#         self.real_bias = nn.Parameter(data = torch.Tensor(out_channels), requires_grad = True) if bias else None

#         self.imag_kernel = nn.Parameter(data = torch.Tensor(out_channels, in_channels, kernel_size, kernel_size), requires_grad = True)
#         self.imag_bias = nn.Parameter(data = torch.Tensor(out_channels), requires_grad = True) if bias else None

#         self.stride, self.padding = stride, padding

#         self.real_kernel.data.uniform_(-0.1, 0.1)
#         self.imag_kernel.data.uniform_(-0.1, 0.1)
#         self.real_bias.data.zero_()
#         self.imag_bias.data.zero_()

#     def to(self, device):
#         self.real_kernel = self.real_kernel.to(device)
#         self.imag_kernel = self.imag_kernel.to(device)
#         return self

#     def forward(self, x):
#         '''
#             x: [bs, c, n, n]
#         '''

#         x_fft = torch.rfft(x, 2, onesided = False)
#         x_fft_real, x_fft_imag = x_fft[..., 0], x_fft[..., 1]

#         x_fft_conv_real1 = nn.functional.conv2d(x_fft_real, self.real_kernel, stride = self.stride, bias = self.real_bias, padding = self.padding)
#         x_fft_conv_real2 = nn.functional.conv2d(x_fft_imag, self.imag_kernel, stride = self.stride, padding = self.padding)
#         x_fft_conv_imag1 = nn.functional.conv2d(x_fft_real, self.imag_kernel, stride = self.stride, bias = self.imag_bias, padding = self.padding)
#         x_fft_conv_imag2 = nn.functional.conv2d(x_fft_imag, self.real_kernel, stride = self.stride, padding = self.padding)

#         x_fft_conv = torch.stack([x_fft_conv_real1 - x_fft_conv_real2, x_fft_conv_imag1 + x_fft_conv_imag2], dim = -1)

#         x_conv = torch.irfft(x_fft_conv, 2, onesided = False)

#         return x_conv

'''
    Fourier convolutional layer 2.0
'''
class FourierConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias = True):
        super(FourierConv2D, self).__init__()

        self.kernel = nn.Parameter(data = torch.Tensor(out_channels, in_channels, kernel_size, kernel_size), requires_grad = True)
        self.kernel.data.uniform_(-0.1, 0.1)

        if bias:
            self.bias = nn.Parameter(data = torch.Tensor(out_channels), requires_grad = True)
            self.bias.data.zero_()
        else:
            self.bias = None

        self.out_channels, self.in_channels, self.kernel_size = out_channels, in_channels, kernel_size

    '''
        2020.10.25
    '''
    def complex_element_wise_product(self, x, y):
        xr, xi = x[..., 0], x[..., 1]
        yr, yi = y[..., 0], y[..., 1]
        return torch.stack([xr * yr - xi * yi, xr * yi + xi * yr], dim = -1)

    def forward(self, img):
        bs, _, h, w = img.size()

        new_h, new_w = h + self.kernel_size - 1, w + self.kernel_size - 1
        pad_h, pad_w = (new_h - h) // 2, (new_w - w) // 2
        
        new_img = F.pad(img, (pad_w, pad_w, pad_h, pad_h), mode = 'constant', value = 0)

        new_kernel = torch.zeros([self.out_channels, self.in_channels, new_h, new_w]).to(self.kernel.get_device())
        k_center = self.kernel_size // 2
        ky, kx = torch.meshgrid(torch.arange(self.kernel_size), torch.arange(self.kernel_size))
        kny = (ky.flip(0) - k_center) % new_h
        knx = (kx.flip(1) - k_center) % new_w
        new_kernel[..., kny, knx] = self.kernel[..., ky, kx]

        new_img_fft = torch.rfft(new_img, 2, onesided = False)
        new_kernel_fft = torch.rfft(torch.transpose(new_kernel, 0, 1), 2, onesided = False)

        res_fft = torch.zeros(bs, self.out_channels, new_h, new_w, 2).to(img.get_device())
        for i in range(self.in_channels):
            res_fft = res_fft + self.complex_element_wise_product(new_img_fft[:, i, ...].unsqueeze(1), new_kernel_fft[i, ...].unsqueeze(0))
        res = torch.irfft(res_fft, 2, onesided = False)

        return res[..., pad_h:-pad_h, pad_w:-pad_w] + self.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)


'''
    Spectral pooling layer
'''
class SpectralPooling2d(nn.Module):
    def __init__(self, kernel_size):
        super(SpectralPooling2d, self).__init__()

        self.kernel_size = 2

    def crop_spectrum(self, z, H, W):
        '''
            z: [bs, c, M, N, 2]
            Return: [bs, c, H, H, 2]
        '''
        M, N = z.size(-3), z.size(-2)
        return z[..., M//2-H//2:M//2+H//2, N//2-W//2:N//2+W//2, :]

    def pad_spectrum(self, z, M, N):
        '''
            z: [bs, c, H, W, 2]
            Return: [bs, c, M, N, 2]
        '''
        H, W = z.size(-3), z.size(-2)
        z_real, z_imag = z[..., 0], z[..., 1]
        pad = torch.nn.ZeroPad2d((N-W)//2, (N-W)//2, (M-H)//2, (M-H)//2)
        return torch.stack([pad(z_real), pad(z_imag)], dim = -1)

    def treat_corner_cases(self, freq_map):
        '''
            freq_map: [bs, c, M, N, 2]
        '''
        S = [(0, 0)]
        M, N = freq_map.size(-3), freq_map.size(-2)

        if M % 2 == 1:
            S.append((M // 2, 0))
        if N % 2 == 1:
            S.append((0, N // 2))
        if M % 2 == 1 and N % 2 == 1:
            S.append((M // 2, N // 2))

        for h, w in S:
            freq_map[..., h, w, 1].zero_()

        return freq_map, S

    def remove_redundancy(self, y):
        '''
            y: input gradient map [bs, c, M, N, 2]
        '''
        z, S = self.treat_corner_cases(y)
        I = []
        M, N = y.size(-3), y.size(-2)

        for m in range(M):
            for n in range(N // 2 + 1):
                if (m, n) not in S:
                    if (m, n) not in I:
                        z[..., m, n, :].mul_(2)
                        I.append((m, n))
                        I.append(((M - m) % M, (N - n) % N))
                    else:
                        z[..., m, n, :].zero_()
        
        return z

    def recover_map(self, y):
        z, S = self.treat_corner_cases(y)
        I = []
        M, N = y.size(-3), y.size(-2)

        for m in range(M):
            for n in range(N // 2 + 1):
                if (m, n) not in S:
                    if (m, n) not in I:
                        z[..., m, n, :].mul_(0.5)
                        z[..., (M-m)%M, (N-n)%N] = z[..., m, n, :]
                        I.append((m, n))
                        I.append(((M - m) % M, (N - n) % N))
                    else:
                        z[..., m, n, :].zero_()

        return z

    def forward(self, x):
        M, N = x.size(-2), x.size(-1)
        H, W = M // self.kernel_size, N // self.kernel_size

        x_fft = torch.rfft(x, 2, onesided = False)
        crop_x_fft = self.crop_spectrum(x_fft, H, W)
        crop_x_fft, _ = self.treat_corner_cases(crop_x_fft)
        pool_x = torch.irfft(crop_x_fft, 2, onesided = False)
        return pool_x

    def backward(self, gRgx):
        H, W = gRgx.size(-2), gRgx.size(-1)
        M, N = H * self.kernel_size, W * self.kernel_size

        z = torch.rfft(gRgx, 2, onesided = False)
        z = self.remove_redundancy(z)
        z = self.pad_spectrum(z, M, N)
        z = self.recover_map(z)
        gRx = torch.irfft(z, 2, onesided = False)

        return gRx

if __name__ == '__main__':
    import time

    fc = FourierConv2D(1, 3, 3)
    # fc = nn.Conv2d(1, 3, 3)
    for param in fc.parameters():
        print(param)

    # with SummaryWriter('./log') as w:
    #     w.add_graph(fc, input_to_model = x)

    optim = torch.optim.Adam(fc.parameters(), lr = 0.1)

    for _ in range(6):
        x = torch.randn([5, 1, 6, 6], requires_grad = True)
        
        optim.zero_grad()
        start = time.time()
        y = fc(x)
        end = time.time()
        loss = torch.sum(y)
        loss.backward()
        optim.step()
        print(end - start)

    for param in fc.parameters():
        print(param)