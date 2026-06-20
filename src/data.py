from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import numpy as np
import torch
import json
import os

dataset_dir = "F:\\Work\\Projects\\VSProjects\\Brain\\datasets\\DVS  Gesture dataset"

history = {
    'train_loss': [],
    'train_acc': [],
    'test_loss': [],
    'test_acc': [],
    'epoch_times': [],
    'train_times': [],
    'test_times': []
}


def create_loaders(T, batch_size, nw=4):
    """
    Создание загрузчиков данных для обучения и тестирования.
    """
    train_set = DVS128Gesture(dataset_dir, train=True, data_type='frame', frames_number=T, split_by='number')
    test_set = DVS128Gesture(dataset_dir, train=False, data_type='frame', frames_number=T, split_by='number')

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=nw)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=nw)

    return train_loader, test_loader


def update_history(train_loss, train_acc, test_loss, test_acc, epoch_time, train_time, test_time):
    """
    Обновление словаря истории обучения новыми значениями.
    """
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['test_loss'].append(test_loss)
    history['test_acc'].append(test_acc)
    history['epoch_times'].append(epoch_time)
    history['train_times'].append(train_time)
    history['test_times'].append(test_time)


def save_history(path, total_time, best_acc, T, batch_size, num_epochs, learning_rate, device):
    """
    Сохранение истории обучения и параметров эксперимента в JSON-файл.
    """
    with open(path, 'w') as f:
        json.dump({
            'history': history,
            'total_time': total_time,
            'best_accuracy': best_acc,
            'parameters': {
                'T': T,
                'batch_size': batch_size,
                'num_epochs': num_epochs,
                'learning_rate': learning_rate,
                'device': device
            }
        }, f, indent=4)


class DVSGestureDataset(Dataset):
    """
    Кастомный класс датасета для загрузки данных DVS Gesture из .npz файлов.
    """

    def __init__(self, root_dir, split='train', transform=None):
        """
        Инициализация датасета: сканирование директории и загрузка списков файлов.
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform

        self.data_dir = self.root_dir / split

        self.samples = []
        self.labels = []

        for label_dir in sorted(os.listdir(self.data_dir)):
            label = int(label_dir)
            label_path = self.data_dir / label_dir

            for file in os.listdir(label_path):
                if file.endswith('.npz'):
                    self.samples.append(label_path / file)
                    self.labels.append(label)

        print(f"Загружено {len(self.samples)} наборов из {split} датасета")
        print(f"Классы: {set(self.labels)}")

    def __len__(self):
        """
        Возвращает общее количество образцов в датасете.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Загружает и возвращает образец по индексу.
        """
        file_path = self.samples[idx]
        label = self.labels[idx]

        data = np.load(file_path)

        frames_key = list(data.keys())[0]
        frames = data[frames_key]

        frames = torch.FloatTensor(frames)
        label = torch.LongTensor([label])[0]

        if self.transform:
            frames = self.transform(frames)

        return frames, label