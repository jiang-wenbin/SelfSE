# model architecture

UNet_8_256 = {
    # (in_channels, out_channels, kernel_size, stride, padding)
    'encoder':
        [[0,  32, (5, 2), (2, 1), (1, 1)],
         (32,  64, (5, 2), (2, 1), (2, 1)),
         (64, 128, (5, 2), (2, 1), (2, 1)),
         (128, 256, (5, 2), (2, 1), (2, 1))],
    # (in_channels, out_channels, kernel_size, stride, padding, output_padding, is_last)
    'decoder':
        [(512, 128, (5, 2), (2, 1), (2, 0), (1, 0)),
         (256,  64, (5, 2), (2, 1), (2, 0), (1, 0)),
         (128,  32, (5, 2), (2, 1), (2, 0), (1, 0)),
         [64,   0, (5, 2), (2, 1), (1, 0), (0, 0), True]]
}

UNet_6_128 = {
    # (in_channels, out_channels, kernel_size, stride, padding)
    'encoder':
        [[0,  32, (5, 2), (2, 1), (1, 1)],
         (32,  64, (5, 2), (2, 1), (2, 1)),
         (64, 128, (5, 2), (2, 1), (2, 1))],
    # (in_channels, out_channels, kernel_size, stride, padding, output_padding, is_last)
    'decoder':
        [(256,  64, (5, 2), (2, 1), (2, 0), (1, 0)),
         (128,  32, (5, 2), (2, 1), (2, 0), (1, 0)),
         [64,   0, (5, 2), (2, 1), (1, 0), (0, 0), True]]
}

UNet_6_64 = {
    # (in_channels, out_channels, kernel_size, stride, padding, causal)
    'encoder':
        [[0,  16, (5, 2), (2, 1), (1, 1)],
         (16, 32, (5, 2), (2, 1), (2, 1)),
         (32, 64, (1, 1), (1, 1), (0, 0), False)],  # 1*1, causal=False
    # (in_channels, out_channels, kernel_size, stride, padding, output_padding, is_last, causal)
    'decoder':
        [(128, 32, (1, 1), (1, 1), (0, 0), (0, 0), False, False),  # 1*1, causal=False
         (64,  16, (5, 2), (2, 1), (2, 0), (1, 0)),
         [32,   0, (5, 2), (2, 1), (1, 0), (0, 0), True]]
}

UNet_6_32 = {
    # (in_channels, out_channels, kernel_size, stride, padding, causal)
    'encoder':
        [[0,  8, (5, 2), (2, 1), (1, 1)],
         (8, 16, (5, 2), (2, 1), (2, 1)),
         (16, 32, (5, 2), (2, 1), (2, 1))],
    # (in_channels, out_channels, kernel_size, stride, padding, output_padding, is_last, causal)
    'decoder':
        [(64, 16, (5, 2), (2, 1), (2, 0), (1, 0)),
         (32,  8, (5, 2), (2, 1), (2, 0), (1, 0)),
         [16,  0, (5, 2), (2, 1), (1, 0), (0, 0), True]]
}
