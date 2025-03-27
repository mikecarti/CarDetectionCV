import cv2
import os

def extract_frames(video_path, output_folder, frame_rate):
    # Создаем выходную папку, если она не существует
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Открываем видеофайл
    cap = cv2.VideoCapture(video_path)

    # Получаем общее количество кадров в видео
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)  # Получаем frame rate видео

    # Вычисляем интервал кадров для извлечения
    frame_interval = int(fps / frame_rate)

    count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Выход из цикла, если видео закончилось

        # Сохраняем кадр, если он соответствует заданному frame rate
        if count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f'frame_{saved_count:04d}.jpg')
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        count += 1

    # Освобождаем ресурсы
    cap.release()
    print(f'Извлечено {saved_count} кадров и сохранено в папку: {output_folder}')

# Пример использования
video_path = '17_50-18_05.mp4'  # Укажите путь к вашему видео
output_folder = 'output_frames'          # Укажите папку для сохранения кадров
frame_rate = 2                      # Укажите желаемый frame rate

extract_frames(video_path, output_folder, frame_rate)