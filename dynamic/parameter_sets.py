from transform_utils import *


class IPN_Parameters:
    def __init__(self, config):
        scales = [1]

        for i in range(1, 5):
            scales.append(scales[-1] * 0.84089641525)

        mean = [114.7748, 107.7354, 99.475]
        std = [38.7568578, 37.88248729, 40.02898126]

        self.train_spatial_transform = Compose(
            [
                MultiScaleRandomCrop(scales, config.sample_size),
                SpatialElasticDisplacement(),
                ToTensor(1),
                Normalize(mean, std),
            ]
        )

        self.train_temporal_transform = Compose(
            [TemporalRandomCrop(config.sample_duration)]
        )

        self.val_spatial_transform = Compose(
            [
                Scale(config.sample_size),
                CenterCrop(config.sample_size),
                ToTensor(1),
                Normalize(mean, std),
            ]
        )

        self.val_temporal_transform = Compose(
            [TemporalCenterCrop(config.sample_duration)]
        )

        self.optimizer_type = "SGD"
        self.lr = 0.001
        self.momentum = 0.9
        self.dampening = 0.9
        self.weight_decay = 0.001


class LSTM_Parameters:
    def __init__(self, config):
        scales = [1]

        for i in range(1, 5):
            scales.append(scales[-1] * 0.84089641525)

        mean = [114.7748, 107.7354, 99.475]
        std = [38.7568578, 37.88248729, 40.02898126]

        self.train_spatial_transform = Compose(
            [
                MultiScaleRandomCrop(scales, config.sample_size),
                SpatialElasticDisplacement(),
                ToTensor(1),
                Normalize(mean, std),
            ]
        )

        self.train_temporal_transform = Compose(
            [TemporalRandomCrop(config.sample_duration)]
        )

        self.val_spatial_transform = Compose(
            [
                Scale(config.sample_size),
                CenterCrop(config.sample_size),
                ToTensor(1),
                Normalize(mean, std),
            ]
        )

        self.val_temporal_transform = Compose(
            [TemporalCenterCrop(config.sample_duration)]
        )

        self.optimizer_type = "SGD"
        self.lr = 0.01
        self.momentum = 0.9
        self.dampening = 0.9
        self.weight_decay = 0.001
