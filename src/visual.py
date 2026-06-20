from torchinfo import summary
import matplotlib.pyplot as plt
import numpy as np
import json
import os


def print_summary(model, input_size):
    """
    Выводит подробную сводку архитектуры нейронной сети.
    Использует библиотеку torchinfo для отображения информации о слоях,
    количестве параметров и размерах тензоров.
    """
    print("\n" + "=" * 140)
    print(" " * 50 + f"Параметры обучения {model.__class__.__name__}")

    summary(
        model,
        input_size=input_size,
        col_names=["input_size", "output_size", "num_params", "trainable"],
        row_settings=["var_names"],
        depth=5,
        verbose=1
    )


def visualize_training_results(json_path):
    """
    Визуализирует результаты обучения из JSON-файла истории.
    Создает комплексный график с тремя подграфиками:
    1. Динамика функции потерь
    2. Динамика точности распознавания
    3. Время выполнения каждой эпохи 
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Извлечение имени модели и функции потерь из пути
    model_name = json_path.split("\\")[1]
    loss_f = json_path.split("\\")[2]

    history = data['history']
    best_acc = data['best_accuracy']
    total_time = data['total_time']
    params = data['parameters']

    # Извлечение данных
    epochs = range(1, len(history['train_loss']) + 1)
    train_loss = history['train_loss']
    train_acc = history['train_acc']
    test_loss = history['test_loss']
    test_acc = history['test_acc']
    epoch_times = history['epoch_times']

    # Создание подграфиков
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f'Результаты обучения {model_name} c функцией ошибки {loss_f}\n'
                 f'Лучшая точность: {best_acc:.2f}% | Время: {total_time / 60:.1f} мин',
                 fontsize=14, fontweight='bold')

    ax1 = plt.subplot(2, 2, 1)       # График потерь
    ax2 = plt.subplot(2, 2, 2)       # График точности
    ax3 = plt.subplot(2, 2, (3, 4))  # График времени эпох (объединенные ячейки)

    # 1. График функции потерь
    ax1.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, test_loss, 'r-', label='Test Loss', linewidth=2)
    ax1.set_xlabel('Эпоха', fontsize=12)
    ax1.set_ylabel('Потери (Loss)', fontsize=12)
    ax1.set_title('Динамика функции потерь', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, len(epochs))
    ax1.yaxis.set_major_locator(plt.MaxNLocator(nbins='auto', integer=False))
    ax1.yaxis.set_minor_locator(plt.NullLocator())

    # 2. График точности
    ax2.plot(epochs, train_acc, 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, test_acc, 'r-', label='Test Accuracy', linewidth=2)
    ax2.axhline(y=best_acc, color='g', linestyle='--', alpha=0.7, label=f'Best: {best_acc:.2f}%')
    ax2.set_xlabel('Эпоха', fontsize=12)
    ax2.set_ylabel('Точность (%)', fontsize=12)
    ax2.set_title('Динамика точности распознавания', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1, len(epochs))
    ax2.set_ylim(0, 100)

    # 3. Столбчатая диаграмма времени выполнения эпох
    ax3.bar(epochs, epoch_times, color='steelblue', alpha=0.7, edgecolor='navy')
    ax3.set_xlabel('Эпоха', fontsize=12)
    ax3.set_ylabel('Время (секунды)', fontsize=12)
    ax3.set_title('Время выполнения одной эпохи', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_xlim(0, len(epochs) + 1)

    plt.tight_layout()

    # Сохранение графика
    output_dir = os.path.dirname(json_path)
    save_path = os.path.join(output_dir, 'training_plots.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📊 График сохранён: {save_path}")

    plt.show()

    # Вывод статистики обучения
    print("\n" + "=" * 60)
    print("СТАТИСТИКА ОБУЧЕНИЯ")
    print("=" * 60)
    print(f"Лучшая точность на тесте:     {best_acc:.2f}%")
    print(f"Лучшая точность на обучении:  {max(train_acc):.2f}%")
    print(f"Минимальная ошибка на тесте:  {min(test_loss):.4f}")
    print(f"Общее время обучения:         {total_time / 60:.1f} минут")
    print(f"Среднее время эпохи:          {np.mean(epoch_times):.1f} секунд")
    print(f"Эпоха с лучшей точностью:     {np.argmax(test_acc) + 1}")

    print("\n" + "=" * 60)
    print(" ГИПЕРПАРАМЕТРЫ")
    print("=" * 60)
    for key, value in params.items():
        print(f"   {key}: {value}")

    return fig


if __name__ == '__main__':
    """
    Точка входа для визуализации результатов обучения.
    """
    # CNN модели
    # visualize_training_results("training_models\\CNN_DVSGesture\\CrossEntropyLoss\\history.json")
    # visualize_training_results("training_models\\CNN_DVSGesture\\MSELoss\\history.json")
    # visualize_training_results("training_models\\CNN_N_DVSGesture\\CrossEntropyLoss\\history.json")
    # visualize_training_results("training_models\\CNN_N_DVSGesture\\MSELoss\\history.json")
    
    # SNN модели
    # visualize_training_results("training_models\\SNN_DVSGesture\\CrossEntropyLoss\\history.json")
    # visualize_training_results("training_models\\SNN_DVSGesture\\MSELoss\\history.json")
    # visualize_training_results("training_models\\SNN_NJ_DVSGesture\\CrossEntropyLoss\\history.json")
    # visualize_training_results("training_models\\SNN_NJ_DVSGesture\\MSELoss\\history.json")
    
    # SNN вниманием
    # visualize_training_results("training_models\\SNN_T_DVSGesture\\CrossEntropyLoss\\history.json")
    # visualize_training_results("training_models\\SNN_T_DVSGesture\\MSELoss\\history.json")
    visualize_training_results("training_models\\SNN_SP_DVSGesture\\MSELoss\\history.json")