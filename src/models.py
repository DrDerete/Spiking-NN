from spikingjelly.activation_based import neuron, surrogate, functional, layer
from spikingjelly.activation_based.layer import SeqToANNContainer
from copy import deepcopy
import torch.nn.functional as F
import torch.nn as nn
import torch


class CNN_DVSGesture(nn.Module):
    """
    Стандартная сверточная нейронная сеть для классификации DVS жестов.
    """

    def __init__(self, num_classes):
        """
        Инициализация слоев CNN архитектуры.
        """
        super().__init__()
        # 3 сверточных слоя
        self.conv = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # классификатор после
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
            # nn.Softmax(dim=1) нужен для MSE функции ошибки
        )

    def forward(self, x):
        """
        Прямой проход через сеть.
        """
        # x: [N, T, C, H, W]
        N, T, C, H, W = x.shape
        # усредняем по времени
        out = []
        for t in range(T):
            frame = self.conv(x[:, t, :, :, :])
            frame = frame.view(N, -1)
            out.append(frame)

        out = torch.stack(out, dim=1)
        out = out.mean(dim=1)

        return self.fc(out)


class CNN_N_DVSGesture(nn.Module):
    """
    Расширенная сверточная нейронная сеть для классификации DVS жестов.
    """

    def __init__(self, num_classes):
        """
        Инициализация расширенной CNN архитектуры.
        """
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        """
        Прямой проход через сеть.
        """
        # x: [N, T, C, H, W]
        N, T, C, H, W = x.shape
        # усредняем по времени
        out = []
        for t in range(T):
            frame = self.conv(x[:, t, :, :, :])
            frame = frame.view(N, -1)
            out.append(frame)

        out = torch.stack(out, dim=1)
        out = out.mean(dim=1)

        return self.fc(out)


class SNN_DVSGesture(nn.Module):
    """
    Базовая спайковая нейронная сеть с LIF-нейронами.
    """

    def __init__(self, T, num_classes):
        """
        Инициализация базовой SNN архитектуры.
        """
        super().__init__()
        self.T = T
        # Сверточные слои с LIF нейронами
        self.conv = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
            nn.MaxPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
            nn.Linear(128, num_classes),
            # neuron.LIFNode(surrogate_function=surrogate.ATan()) это надо для mse
        )

    def forward(self, x):
        """
        Прямой проход через сеть с разверткой по времени.
        """
        # x: [N, T, C, H, W] из DataLoader
        x = x.permute(1, 0, 2, 3, 4)  # [T, N, C, H, W]

        outputs = []
        for t in range(x.shape[0]):
            x_t = self.conv(x[t])
            x_t = self.fc(x_t)
            outputs.append(x_t)

        # outputs: список из T тензоров [N, num_classes]
        x = torch.stack(outputs, dim=0)  # [T, N, num_classes]

        return x.mean(dim=0)

    def reset_states(self):
        """
        Сброс внутренних состояний всех LIF-нейронов сети.
        """
        functional.reset_net(self)


class SNN_J_DVSGesture(nn.Module):
    """
    Спайковая нейронная сеть с голосованнием на выходе.
    """

    def __init__(self, channels=128, spiking_neuron: callable = None, **kwargs):
        """
        Инициализация SNN с VotingLayer.
        """
        super().__init__()

        conv = []

        for i in range(5):
            in_channels = 2 if i == 0 else channels
            conv.append(layer.Conv2d(in_channels, channels, 3, padding=1, bias=False))
            conv.append(layer.BatchNorm2d(channels))
            conv.append(spiking_neuron(**deepcopy(kwargs)))
            conv.append(layer.MaxPool2d(2, 2))

        self.conv_fc = nn.Sequential(
            *conv,
            layer.Flatten(),  # [N, channels*4*4]
            layer.Dropout(0.5),
            layer.Linear(channels * 4 * 4, 512),  # 128*16 = 2048 → 512
            spiking_neuron(**deepcopy(kwargs)),

            layer.Dropout(0.5),
            layer.Linear(512, 110),  # 110 = 11 классов × 10 голосов
            spiking_neuron(**deepcopy(kwargs)),

            layer.VotingLayer(10)  # голосование → 11 классов
        )

    def forward(self, x: torch.Tensor):
        """
        Прямой проход через сеть.
        """
        return self.conv_fc(x)

    def reset_mem(self):
        """
        Сброс внутренних состояний всех LIF-нейронов сети.
        """
        functional.reset_net(self)


class SNN_NJ_DVSGesture(nn.Module):
    """
    Спайковая нейронная сеть с расширенной архитектурой.
    """

    def __init__(self, T, num_classes):
        """
        Инициализация расширенной SNN архитектуры.
        """
        super().__init__()
        self.T = T

        self.conv = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
            nn.MaxPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 256, bias=False),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes, bias=False),
            # neuron.LIFNode(surrogate_function=surrogate.ATan()) для mse
        )

    def forward(self, x):
        """
        Прямой проход через сеть с разверткой по времени.
        """
        x = x.permute(1, 0, 2, 3, 4)

        spk_rec = []
        for t in range(self.T):
            spk = self.conv(x[t])
            spk = self.fc(spk)
            spk_rec.append(spk)

        return torch.stack(spk_rec).mean(dim=0)

    def reset_states(self):
        """
        Сброс внутренних состояний всех LIF-нейронов сети.
        """
        functional.reset_net(self)


class SNN_T_DVSGesture(nn.Module):
    """
    Спайковая нейронная сеть с пространственным вниманием.
    """

    def __init__(self, T, num_classes):
        """
        Инициализация SNN с механизмом внимания.
        """
        super().__init__()
        self.T = T

        self.conv = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
            nn.MaxPool2d(2),
        )

        self.attention = SSA(dim=128, num_heads=8)

        self.after_attn = nn.Sequential(
            nn.Linear(128, 128),
            neuron.LIFNode(surrogate_function=surrogate.ATan())
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 256),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
            # neuron.LIFNode(surrogate_function=surrogate.ATan())  # для mse
        )

    def forward(self, x):
        """
        Прямой проход через сеть с пространственным вниманием.
        """
        x = x.permute(1, 0, 2, 3, 4)  # [T, N, C, H, W]

        spk_rec = []
        for t in range(self.T):
            spk = self.conv(x[t])

            N, C, H, W = spk.shape
            spatial_features = spk.flatten(2, 3).permute(0, 2, 1)  # [N, 16, 128]
            spatial_features = spatial_features.unsqueeze(0)  # [1, N, 16, 128]
            attended = self.attention(spatial_features)  # [1, N, 16, 128]
            attended = attended.squeeze(0)  # [N, 16, 128]
            attended = attended.permute(0, 2, 1).reshape(N, C, H, W)  # [N, 128, 4, 4]
            attended = attended.permute(0, 2, 3, 1)  # [N, 4, 4, 128]
            attended = self.after_attn(attended)  # [N, 4, 4, 128]
            attended = attended.permute(0, 3, 1, 2)  # [N, 128, 4, 4]

            spk = attended
            spk = self.fc(spk)
            spk_rec.append(spk)

        return torch.stack(spk_rec).mean(dim=0)

    def reset_states(self):
        """
        Сброс внутренних состояний всех LIF-нейронов сети.
        """
        functional.reset_net(self)


class SSA(nn.Module):
    """
    Спайковый слой самовнимания.
    """

    def __init__(self, dim, num_heads=8):
        """
        Инициализация слоя спайкового самовнимания.
        """
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 0.125

        self.q = self.init_m_block(dim)
        self.k = self.init_m_block(dim)
        self.v = self.init_m_block(dim)

        self.attn_lif = neuron.LIFNode(step_mode="m")

        self.proj_linear = nn.Linear(dim, dim)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = neuron.LIFNode(step_mode="m")

    def forward(self, x):
        """
        Прямой проход через слой внимания.
        """
        T, B, N, C = x.shape  # тайминг, батч, flat(H, W), каналы

        x_flat = x.flatten(0, 1)  # TB, N, C

        q = self.to_multi_head(self.apply_block(self.q, x_flat), T, B, N)
        k = self.to_multi_head(self.apply_block(self.k, x_flat), T, B, N)
        v = self.to_multi_head(self.apply_block(self.v, x_flat), T, B, N)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        x = attn @ v
        x = self._merge_heads(x, T, B, N, C)
        x = self.attn_lif(x)
        x = self._apply_projection(x, T, B, N, C)

        return x

    def _merge_heads(self, x, T, B, N, C):
        """
        Преобразует тензор из [TB, N, C] в multi-head формат [T, B, H, N, D].
        """
        return x.transpose(2, 3).reshape(T, B, N, C).contiguous()

    def _apply_projection(self, x, T, B, N, C):
        """
        Объединяет головы обратно в исходное пространство: [T, B, H, N, D] → [T, B, N, C].
        """
        x = x.flatten(0, 1)
        x = self.proj_linear(x)
        x = self.proj_bn(x.transpose(-1, -2)).transpose(-1, -2)
        x = self.proj_lif(x.reshape(T, B, N, C))
        return x

    def init_m_block(self, dim):
        """
        Инициализация блока Linear + BatchNorm + LIFNode для multi-head обработки.
        """
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            neuron.LIFNode(step_mode="m")
        )

    def apply_block(self, block, x_flat):
        """
        Применение блока преобразования к входному тензору.
        """
        x = block[0](x_flat)  # Linear
        x = block[1](x.transpose(-1, -2)).transpose(-1, -2)  # BatchNorm
        x = block[2](x)  # LIF
        return x

    def to_multi_head(self, tensor, T, B, N):
        """
        Преобразование тензора в multi-head формат.
        """
        return tensor.reshape(T, B, N, self.num_heads, self.head_dim) \
            .permute(0, 1, 3, 2, 4) \
            .contiguous()


class SNN_SP_DVSGesture(nn.Module):
    """
    Спайковая нейронная сеть с пространственно-временным спайковым вниманием.
    """

    def __init__(self, T, num_classes):
        """
        Инициализация.
        """
        super().__init__()
        self.T = T

        self.conv = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            neuron.LIFNode(surrogate_function=surrogate.ATan(), step_mode="m"),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            neuron.LIFNode(surrogate_function=surrogate.ATan(), step_mode="m"),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            neuron.LIFNode(surrogate_function=surrogate.ATan(), step_mode="m"),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            neuron.LIFNode(surrogate_function=surrogate.ATan(), step_mode="m"),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            neuron.LIFNode(surrogate_function=surrogate.ATan(), step_mode="m"),
            nn.MaxPool2d(2),
        )

        self.attention = SpikingSelfAttention(dim=128, num_heads=8)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 256, bias=False),
            neuron.LIFNode(surrogate_function=surrogate.ATan(), step_mode="m"),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes, bias=False),
            neuron.LIFNode(surrogate_function=surrogate.ATan())  # для mse
        )

    def forward(self, x):
        """
        Прямой проход.
        """
        x = x.permute(1, 0, 2, 3, 4)  # [T, B, 2, 128, 128]

        spatial_features = []
        for t in range(self.T):
            spk = self.conv(x[t])
            N, C, H, W = spk.shape
            spk_seq = spk.flatten(2)  # [B, 128, 16]
            spatial_features.append(spk_seq)

        # [T, B, C, N_patches]
        spatial_features = torch.stack(spatial_features)  # [T, B, 128, 16]

        attended = self.attention(spatial_features)  # [T, B, 128, 16]

        outputs = []
        for t in range(self.T):
            feat = attended[t]
            feat = feat.reshape(feat.shape[0], 128, 4, 4)
            out = self.fc(feat)
            outputs.append(out)

        return torch.stack(outputs).mean(dim=0)

    def reset_states(self):
        """
        Сброс внутренних состояний всех LIF-нейронов сети.
        """
        functional.reset_net(self)


class SpikingSelfAttention(nn.Module):
    """
    Эффективный спайковый слой самовнимания
    """

    def __init__(self, dim, num_heads=8):
        """
        Инициализация эффективного слоя спайкового самовнимания.
        """
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 0.125

        self.qkv_conv_bn = SeqToANNContainer(
            nn.Conv1d(dim, dim * 3, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(dim * 3),
        )
        self.qkv_lif = neuron.LIFNode(step_mode="m")

        self.attn_lif = neuron.LIFNode(step_mode="m")

        self.proj_conv_bn = SeqToANNContainer(
            nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(dim),
        )
        self.proj_lif = neuron.LIFNode(step_mode="m")

    def forward(self, x_seq: torch.Tensor):
        """
        Прямой проход через слой спайкового внимания.
        """
        T, B, C, N = x_seq.shape

        qkv = self.qkv_conv_bn(x_seq)
        qkv = self.qkv_lif(qkv)  # [T, B, 3*C, N]
        qkv = qkv.reshape(T, B, 3 * self.num_heads, C // self.num_heads, N)

        qt, kt, vt = qkv.chunk(3, dim=2)
        # qt, kt, vt.shape = [T, B, NUM_HEADS, C//NUM_HEADS, N]
        x_seq = vt @ kt.transpose(-2, -1)
        x_seq = (x_seq @ qt) * self.scale  # [T, B, NUM_HEADS, C//NUM_HEADS, N]

        x_seq = self.attn_lif(x_seq).reshape(T, B, C, N)

        x_seq = self.proj_conv_bn(x_seq)
        x_seq = self.proj_lif(x_seq)  # [T, B, C, N]
        return x_seq