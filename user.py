if __name__ == '__main__':
    pass
    # --- ЗАГРУЗКА ДАННЫХ ---
    x, y = data_prepare('datapath')  # подготовка датасета

    # --- ПОСТРОЕНИЕ МОДЕЛЕЙ ---
    # --- МЕТРИКИ ---

    metric1 = Metric(x_train, y_train)
    # --- ГЕНЕРАЦИЯ ОТЧЕТА ---


