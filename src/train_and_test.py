from spikingjelly.activation_based import functional, neuron
import torch.nn as nn
import torch.nn.functional as F
import models
import visual
import torch
import time
import data
import os

T = 16
batch_size = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_net(model, lr, num_epochs, loss_fn, save):
    """
    Основная функция обучения нейронной сети.
    Выполняет полный цикл обучения модели на заданное количество эпох,
    отслеживает лучшую точность, сохраняет историю обучения и состояния модели.
    """
    name_net = model.__class__.__name__
    save_dir = f"training_models/{name_net}"

    print(f"Результаты будут сохранены в: {save_dir}")

    train_loader, test_loader = data.create_loaders(T=T, batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    save_dir += f"/{loss_fn.__class__.__name__}"
    os.makedirs(save_dir, exist_ok=True)

    total_start_time = time.time()
    best_acc = 0

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        train_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn)
        train_time = time.time() - train_start

        test_start = time.time()
        test_loss, test_acc = test_model(model, test_loader, loss_fn)
        test_time = time.time() - test_start

        epoch_time = time.time() - epoch_start_time

        data.update_history(train_loss, train_acc, test_loss, test_acc, epoch_time, train_time, test_time)

        if test_acc > best_acc:
            best_acc = test_acc
            best_model_path = f"{save_dir}/best_model.pth"
            if save:
                save_model_state(best_model_path, epoch, model.state_dict(), optimizer.state_dict(),
                                 train_acc, test_acc, train_loss, test_loss)

        print(f"Epoch [{epoch + 1:2d}/{num_epochs}] | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:5.2f}% | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:5.2f}% | "
              f"Time: {epoch_time:4.1f}s (T:{train_time:3.1f}s | Ts:{test_time:3.1f}s)")

    total_time = time.time() - total_start_time
    save_history_path = f"{save_dir}/history.json"
    if save:
        data.save_history(save_history_path, total_time, best_acc, T, batch_size, num_epochs, lr, str(device))

    final_model_path = f"{save_dir}/final_model.pth"
    if save:
        save_model_state(final_model_path, epoch, model.state_dict(), optimizer.state_dict(),
                         train_acc, test_acc, train_loss, test_loss)


def train_one_epoch(model, loader, optimizer, loss_fn):
    """
    Выполняет одну эпоху обучения модели.
    Проходит по всем батчам обучающей выборки, обновляет веса модели
    и вычисляет среднюю потерю и точность за эпоху.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for frames, labels in loader:
        frames = frames.to(device)  # [N, T, C, H, W]
        labels = labels.to(device)

        optimizer.zero_grad()
        if hasattr(model, 'reset_states'):
            model.reset_states()

        outputs = model(frames)  # возвращает [N, 11]

        if isinstance(loss_fn, nn.MSELoss):
            loss = loss_fn(outputs, F.one_hot(labels, 11).float())
        else:
            loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    accuracy = 100.0 * correct / total
    return running_loss / len(loader), accuracy


def test_model(model, loader, loss_fn):
    """
    Выполняет тестирование модели на отложенной выборке.
    Оценивает качество модели на тестовых данных без обновления весов.
    """
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0

    with torch.no_grad():
        for frames, labels in loader:
            frames = frames.to(device)
            labels = labels.to(device)
            if hasattr(model, 'reset_states'):
                model.reset_states()

            outputs = model(frames)
            if isinstance(loss_fn, nn.MSELoss):
                loss = loss_fn(outputs, F.one_hot(labels, 11).float())
            else:
                loss = loss_fn(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total
    return running_loss / len(loader), accuracy


def save_model_state(path, epoch, model_state, optimizer_state, train_acc, test_acc, train_loss, test_loss):
    """
    Сохраняет состояние модели и оптимизатора в файл.
    """
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer_state,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'train_loss': train_loss,
        'test_loss': test_loss
    }, path)


def main():
    """
    Основная функция для запуска обучения и тестирования.
    """
    model = models.CNN_DVSGesture(num_classes=11).to(device)
    # model = models.CNN_N_DVSGesture(num_classes=11).to(device)
    # model = models.SNN_DVSGesture(T=T, num_classes=11).to(device)
    # model = models.SNN_NJ_DVSGesture(T=T, num_classes=11).to(device)
    # model = models.SNN_T_DVSGesture(T=T, num_classes=11).to(device)
    # model = models.SNN_SP_DVSGesture(T=T, num_classes=11).to(device)
    # model = models.SNN_J_DVSGesture(channels=128, spiking_neuron=neuron.LIFNode,
    #                                  surrogate_function=surrogate.ATan(), detach_reset=True)

    functional.set_step_mode(model, 'm')
    functional.set_backend(model, 'cupy', instance=neuron.LIFNode)

    # loss = nn.CrossEntropyLoss()
    loss = nn.MSELoss()

    # train_net(model, lr=0.001, num_epochs=100, loss_fn=loss, save=True)

    visual.print_summary(model, [batch_size, T, 2, 128, 128])


if __name__ == '__main__':
    main()