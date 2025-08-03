# unet3_pytorch.py
# PyTorch implementation of the 3‑D U‑Net model with bias terms enabled
# in every convolution (parity with Keras / tf.keras).

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super().__init__()

        # --------------------------------------------------
        # helpers
        # --------------------------------------------------
        def init_weights(m):
            """
            Xavier/Glorot initialisation, zero‑bias.
            """
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        def double_conv(in_c: int, out_c: int) -> nn.Sequential:
            """
            Two consecutive 3×3×3 convolutions with ReLU.
            NOTE: bias=True for equivalence with the Keras versions.
            """
            return nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_c, out_c, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
            )

        # --------------------------------------------------
        # encoder
        # --------------------------------------------------
        self.conv1 = double_conv(in_channels, 16)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv2 = double_conv(16, 32)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv3 = double_conv(32, 64)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        # --------------------------------------------------
        # bottleneck
        # --------------------------------------------------
        self.conv4 = double_conv(64, 128)

        # --------------------------------------------------
        # decoder
        # --------------------------------------------------
        # UpSampling3D in Keras defaults to nearest‑neighbour;
        # we mirror that with nn.Upsample(mode='nearest').
        self.up5   = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv5 = double_conv(128 + 64, 64)

        self.up6   = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv6 = double_conv(64 + 32, 32)

        self.up7   = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv7 = double_conv(32 + 16, 16)

        # --------------------------------------------------
        # output head
        # --------------------------------------------------
        # Explicit bias=True for clarity (default is already True).
        self.conv8 = nn.Conv3d(16, out_channels, kernel_size=1, bias=True)

        # --------------------------------------------------
        # initialise weights
        # --------------------------------------------------
        self.apply(init_weights)

    # ------------------------------------------------------
    # forward pass
    # ------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # encoder
        c1 = self.conv1(x)
        p1 = self.pool1(c1)

        c2 = self.conv2(p1)
        p2 = self.pool2(c2)

        c3 = self.conv3(p2)
        p3 = self.pool3(c3)

        # bottleneck
        c4 = self.conv4(p3)

        # decoder
        u5 = self.up5(c4)
        u5 = torch.cat([u5, c3], dim=1)
        c5 = self.conv5(u5)

        u6 = self.up6(c5)
        u6 = torch.cat([u6, c2], dim=1)
        c6 = self.conv6(u6)

        u7 = self.up7(c6)
        u7 = torch.cat([u7, c1], dim=1)
        c7 = self.conv7(u7)

        # output
        out = torch.sigmoid(self.conv8(c7))
        return out


# -------------------------------------------------------------------------
# factory function (kept for API parity with the Keras / TF versions)
# -------------------------------------------------------------------------
def unet(input_size=None, pretrained_weights=None):
    model = UNet(in_channels=1, out_channels=1)
    if pretrained_weights is not None:
        model.load_state_dict(torch.load(pretrained_weights, map_location="cpu"))
    return model


# -------------------------------------------------------------------------
# balanced cross‑entropy loss (unchanged)
# -------------------------------------------------------------------------
def cross_entropy_balanced(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    PyTorch implementation of the balanced cross‑entropy loss used
    in the Keras / TF versions.  Takes probabilities as input.
    """
    _eps = 1e-7
    y_pred = torch.clamp(y_pred, _eps, 1.0 - _eps)
    logits = torch.log(y_pred / (1.0 - y_pred))

    y_true = y_true.to(torch.float32)

    count_neg = torch.sum(1.0 - y_true)
    count_pos = torch.sum(y_true)

    beta = count_neg / (count_neg + count_pos + _eps)
    pos_weight = beta / (1.0 - beta + _eps)

    loss = F.binary_cross_entropy_with_logits(
        logits, y_true, pos_weight=pos_weight, reduction="none"
    )
    loss = torch.mean(loss * (1.0 - beta))

    return torch.where(torch.eq(count_pos, 0.0), torch.tensor(0.0, device=loss.device), loss)
