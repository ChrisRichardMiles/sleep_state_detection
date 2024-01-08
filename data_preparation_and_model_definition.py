# ## Feature creation

import shutil
from pathlib import Path

import hydra
import numpy as np
import polars as pl
from tqdm import tqdm

from src.conf import PrepareDataConfig
from src.utils.common import trace

SERIES_SCHEMA = {
    "series_id": pl.Utf8,
    "step": pl.UInt32,
    "anglez": pl.Float32,
    "enmo": pl.Float32,
}


FEATURE_NAMES = [
    "anglez",
    "enmo",
    "step",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "minute_sin",
    "minute_cos",
    "anglez_sin",
    "anglez_cos",
    'anglez_diff', 
    'enmo_diff', 
    'anglez_diff_rolling_median', 
    'enmo_diff_rolling_median', 
    'anglez_diff_rolling_median_reverse', 
    'enmo_diff_rolling_median_reverse', 
]

ANGLEZ_MEAN = -8.810476
ANGLEZ_STD = 35.521877
ENMO_MEAN = 0.041315
ENMO_STD = 0.101829


def to_coord(x: pl.Expr, max_: int, name: str) -> list[pl.Expr]:
    rad = 2 * np.pi * (x % max_) / max_
    x_sin = rad.sin()
    x_cos = rad.cos()

    return [x_sin.alias(f"{name}_sin"), x_cos.alias(f"{name}_cos")]


def deg_to_rad(x: pl.Expr) -> pl.Expr:
    return np.pi / 180 * x


def add_feature(series_df: pl.DataFrame) -> pl.DataFrame:
    series_df = (
        series_df.with_row_count("step")
        .with_columns(
            *to_coord(pl.col("timestamp").dt.hour(), 24, "hour"),
            *to_coord(pl.col("timestamp").dt.month(), 12, "month"),
            *to_coord(pl.col("timestamp").dt.minute(), 60, "minute"),
            pl.col("step") / pl.count("step"),
            pl.col('anglez_rad').sin().alias('anglez_sin'),
            pl.col('anglez_rad').cos().alias('anglez_cos'),
            pl.col('anglez').diff().fill_null(0).alias('anglez_diff'), 
            pl.col('enmo').diff().fill_null(0).alias('enmo_diff'), 
            pl.col('anglez').diff().fill_null(0).rolling_median(5 * 12).alias('anglez_diff_rolling_median'), 
            pl.col('enmo').diff().fill_null(0).rolling_median(5 * 12).alias('enmo_diff_rolling_median'),
            pl.col('anglez').diff().fill_null(0).reverse().rolling_median(5 * 12).alias('anglez_diff_rolling_median_reverse'), 
            pl.col('enmo').diff().fill_null(0).reverse().rolling_median(5 * 12).alias('enmo_diff_rolling_median_reverse'),
        )
        .select("series_id", *FEATURE_NAMES)
    )
    fill_cols = [
        'anglez_diff_rolling_median', 
        'enmo_diff_rolling_median', 
        'anglez_diff_rolling_median_reverse', 
        'enmo_diff_rolling_median_reverse',
    ]
    for col in fill_cols:
        mean_value = series_df.select(pl.col(col).mean()).to_numpy()[0, 0]
        series_df = series_df.with_columns(pl.col(col).fill_null(pl.lit(mean_value)))
    return series_df


def save_each_series(this_series_df: pl.DataFrame, columns: list[str], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    for col_name in columns:
        x = this_series_df.get_column(col_name).to_numpy(zero_copy_only=True)
        np.save(output_dir / f"{col_name}.npy", x)


@hydra.main(config_path="conf", config_name="prepare_data", version_base="1.2")
def main(cfg: PrepareDataConfig):
    processed_dir: Path = Path(cfg.dir.processed_dir) / cfg.phase

    # ディレクトリが存在する場合は削除
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
        print(f"Removed {cfg.phase} dir: {processed_dir}")

    with trace("Load series"):
        # scan parquet
        if cfg.phase in ["train", "test"]:
            series_lf = pl.scan_parquet(
                Path(cfg.dir.data_dir) / f"{cfg.phase}_series.parquet",
                low_memory=True,
            )
        elif cfg.phase == "dev":
            series_lf = pl.scan_parquet(
                Path(cfg.dir.processed_dir) / f"{cfg.phase}_series.parquet",
                low_memory=True,
            )
        else:
            raise ValueError(f"Invalid phase: {cfg.phase}")

        # preprocess
        series_df = (
            series_lf.with_columns(
                pl.col("timestamp").str.to_datetime("%Y-%m-%dT%H:%M:%S%z"),
                deg_to_rad(pl.col("anglez")).alias("anglez_rad"),
                (pl.col("anglez") - ANGLEZ_MEAN) / ANGLEZ_STD,
                (pl.col("enmo") - ENMO_MEAN) / ENMO_STD,
            )
            .select(
                [
                    pl.col("series_id"),
                    pl.col("anglez"),
                    pl.col("enmo"),
                    pl.col("timestamp"),
                    pl.col("anglez_rad"),
                ]
            )
            .collect(streaming=True)
            .sort(by=["series_id", "timestamp"])
        )
        n_unique = series_df.get_column("series_id").n_unique()
    with trace("Save features"):
        for series_id, this_series_df in tqdm(series_df.group_by("series_id"), total=n_unique):
            # 特徴量を追加
            this_series_df = add_feature(this_series_df)

            # 特徴量をそれぞれnpyで保存
            series_dir = processed_dir / series_id  # type: ignore
            save_each_series(this_series_df, FEATURE_NAMES, series_dir)


if __name__ == "__main__":
    main()

# ## Model Encoder

# +
from typing import Optional

import torch
import torch.nn as nn


class LSTMFeatureExtractor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        num_layers: int,
        bidirectional: bool,
        stride: int,
        out_size: Optional[int] = None,
    ):
        super().__init__()
        self.fc = nn.Linear(in_channels, hidden_size)
        self.conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=stride,
            stride=stride,
            padding=1,
        )
        self.height = hidden_size * (2 if bidirectional else 1)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.out_chans = 1
        self.out_size = out_size
        if self.out_size is not None:
            self.pool = nn.AdaptiveAvgPool2d((None, self.out_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Args:
            x (torch.Tensor): (batch_size, in_channels, time_steps)

        Returns:
            torch.Tensor: (batch_size, out_chans, height, time_steps)
        """
        # x: (batch_size, in_channels, time_steps)
        x = self.fc(x.transpose(1, 2))  # x: (batch_size, time_steps, hidden_size)
        x = self.conv(x.transpose(1, 2))  # x: (batch_size, hidden_size, time_steps)

        if self.out_size is not None:
            x = x.unsqueeze(1)  # x: (batch_size, 1, hidden_size, time_steps)
            x = self.pool(x)  # x: (batch_size, 1, hidden_size, output_size)
            x = x.squeeze(1)  # x: (batch_size, hidden_size, output_size)
        x = x.transpose(1, 2)  # x: (batch_size, output_size, hidden_size)
        x, _ = self.lstm(x)  # x: (batch_size, output_size, hidden_size * num_directions)
        x = x.transpose(1, 2)  # x: (batch_size, hidden_size * num_directions, output_size)
        x = x.unsqueeze(1)  # x: (batch_size, out_chans, hidden_size * num_directions, time_steps)
        return x


# -

# ## Decoder 

# +
# ref: https://github.com/bamps53/kaggle-dfl-3rd-place-solution/blob/master/models/cnn_3d.py
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEModule(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        (
            b,
            c,
            _,
        ) = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(
        self,
        in_channels,
        out_channels,
        mid_channels=None,
        norm=nn.BatchNorm1d,
        se=False,
        res=False,
    ):
        super().__init__()
        self.res = res
        if not mid_channels:
            mid_channels = out_channels
        if se:
            non_linearity = SEModule(out_channels)
        else:
            non_linearity = nn.ReLU(inplace=True)
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            norm(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm(out_channels),
            non_linearity,
        )

    def forward(self, x):
        if self.res:
            return x + self.double_conv(x)
        else:
            return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(
        self, in_channels, out_channels, scale_factor, norm=nn.BatchNorm1d, se=False, res=False
    ):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(scale_factor),
            DoubleConv(in_channels, out_channels, norm=norm, se=se, res=res),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(
        self, in_channels, out_channels, bilinear=True, scale_factor=2, norm=nn.BatchNorm1d
    ):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, norm=norm)
        else:
            self.up = nn.ConvTranspose1d(
                in_channels, in_channels // 2, kernel_size=scale_factor, stride=scale_factor
            )
            self.conv = DoubleConv(in_channels, out_channels, norm=norm)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diff = x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, [diff // 2, diff - diff // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


def create_layer_norm(channel, length):
    return nn.LayerNorm([channel, length])


class UNet1DDecoder(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        duration: int,
        bilinear: bool = True,
        se: bool = False,
        res: bool = False,
        scale_factor: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.duration = duration
        self.bilinear = bilinear
        self.se = se
        self.res = res
        self.scale_factor = scale_factor

        factor = 2 if bilinear else 1
        self.inc = DoubleConv(
            self.n_channels, 64, norm=partial(create_layer_norm, length=self.duration)
        )
        self.down1 = Down(
            64, 128, scale_factor, norm=partial(create_layer_norm, length=self.duration // 2)
        )
        self.down2 = Down(
            128, 256, scale_factor, norm=partial(create_layer_norm, length=self.duration // 4)
        )
        self.down3 = Down(
            256, 512, scale_factor, norm=partial(create_layer_norm, length=self.duration // 8)
        )
        self.down4 = Down(
            512,
            1024 // factor,
            scale_factor,
            norm=partial(create_layer_norm, length=self.duration // 16),
        )
        self.up1 = Up(
            1024,
            512 // factor,
            bilinear,
            scale_factor,
            norm=partial(create_layer_norm, length=self.duration // 8),
        )
        self.up2 = Up(
            512,
            256 // factor,
            bilinear,
            scale_factor,
            norm=partial(create_layer_norm, length=self.duration // 4),
        )
        self.up3 = Up(
            256,
            128 // factor,
            bilinear,
            scale_factor,
            norm=partial(create_layer_norm, length=self.duration // 2),
        )
        self.up4 = Up(
            128, 64, bilinear, scale_factor, norm=partial(create_layer_norm, length=self.duration)
        )

        self.cls = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, self.n_classes, kernel_size=1, padding=0),
            nn.Dropout(dropout),
        )
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self, x: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> dict[str, Optional[torch.Tensor]]:
        """Forward

        Args:
            x (torch.Tensor): (batch_size, n_channels, n_timesteps)

        Returns:
            torch.Tensor: (batch_size, n_timesteps, n_classes)
        """

        # 1D U-Net
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # classifier
        logits = self.cls(x)  # (batch_size, n_classes, n_timesteps)
        return logits.transpose(1, 2)  # (batch_size, n_timesteps, n_classes)

