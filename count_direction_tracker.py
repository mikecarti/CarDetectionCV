import cv2
from ultralytics import YOLO
import csv
import time
from tqdm import tqdm
import os
import warnings

# Отключаем вывод YOLO
warnings.filterwarnings("ignore", category=UserWarning)

class DirectionCounter:
    def __init__(self, frame_width, frame_height):
        self.left_line = frame_width // 4
        self.right_line = 9 * frame_width // 10
        self.top_line = frame_height // 3
        self.bottom_line = 5 * frame_height // 6
        
        self.directions = {
            'west': {'in': 0, 'out': 0},
            'east': {'in': 0, 'out': 0},
            'north': {'in': 0, 'out': 0},
            'south': {'in': 0, 'out': 0}
        }
        
        self.prev_positions = {}
        self.tracked_entries = set()
        self.tracked_exits = set()

    def update_counters(self, track_id, center_x, center_y):
        if track_id not in self.prev_positions:
            self.prev_positions[track_id] = (center_x, center_y)
            return

        prev_x, prev_y = self.prev_positions[track_id]
        
        self._check_west(prev_x, center_x, track_id)
        self._check_east(prev_x, center_x, track_id)
        self._check_north(prev_y, center_y, track_id)
        self._check_south(prev_y, center_y, track_id)
        
        self.prev_positions[track_id] = (center_x, center_y)

    def _check_west(self, prev_x, curr_x, track_id):
        if prev_x > self.left_line and curr_x <= self.left_line:
            if track_id not in self.tracked_entries:
                self.directions['west']['in'] += 1
                self.tracked_entries.add(track_id)
        elif prev_x < self.left_line and curr_x >= self.left_line:
            if track_id not in self.tracked_exits:
                self.directions['west']['out'] += 1
                self.tracked_exits.add(track_id)

    def _check_east(self, prev_x, curr_x, track_id):
        if prev_x < self.right_line and curr_x >= self.right_line:
            if track_id not in self.tracked_entries:
                self.directions['east']['in'] += 1
                self.tracked_entries.add(track_id)
        elif prev_x > self.right_line and curr_x <= self.right_line:
            if track_id not in self.tracked_exits:
                self.directions['east']['out'] += 1
                self.tracked_exits.add(track_id)

    def _check_north(self, prev_y, curr_y, track_id):
        if prev_y > self.top_line and curr_y <= self.top_line:
            if track_id not in self.tracked_entries:
                self.directions['north']['in'] += 1
                self.tracked_entries.add(track_id)
        elif prev_y < self.top_line and curr_y >= self.top_line:
            if track_id not in self.tracked_exits:
                self.directions['north']['out'] += 1
                self.tracked_exits.add(track_id)

    def _check_south(self, prev_y, curr_y, track_id):
        if prev_y < self.bottom_line and curr_y >= self.bottom_line:
            if track_id not in self.tracked_entries:
                self.directions['south']['in'] += 1
                self.tracked_entries.add(track_id)
        elif prev_y > self.bottom_line and curr_y <= self.bottom_line:
            if track_id not in self.tracked_exits:
                self.directions['south']['out'] += 1
                self.tracked_exits.add(track_id)

    def draw_lines(self, frame):
        cv2.line(frame, (self.left_line, 0), (self.left_line, frame.shape[0]), (0, 255, 0), 2)
        cv2.line(frame, (self.right_line, 0), (self.right_line, frame.shape[0]), (0, 255, 0), 2)
        cv2.line(frame, (0, self.top_line), (frame.shape[1], self.top_line), (0, 255, 0), 2)
        cv2.line(frame, (0, self.bottom_line), (frame.shape[1], self.bottom_line), (0, 255, 0), 2)
        return frame

def process_video(video_path, model, output_dir="results", show_video=False, save_video=False):
    # Создаем папку для результатов, если ее нет
    os.makedirs(output_dir, exist_ok=True)
    
    # Получаем имя файла без расширения
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    csv_path = os.path.join(output_dir, f"{video_name}_counts.csv")
    
    # Открытие видео
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Ошибка: Не удалось открыть видео {video_path}.")
        return None

    # Получаем параметры видео
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Инициализация счетчика направлений
    direction_counter = DirectionCounter(frame_width, frame_height)

    # Инициализация VideoWriter если нужно сохранять видео
    if save_video:
        video_output_path = os.path.join(output_dir, f"{video_name}_processed.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_output_path, fourcc, fps, (frame_width, frame_height))
    else:
        out = None

    # Создаем progress bar
    pbar = tqdm(total=total_frames, desc=f"Обработка {video_name}", unit="кадр")

    # Обработка видео
    processed_frames = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Трекинг объектов с подавлением вывода
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
        
        # Подготовка кадра для отображения/сохранения
        if show_video or save_video:
            annotated_frame = frame.copy()
            annotated_frame = direction_counter.draw_lines(annotated_frame)
        else:
            annotated_frame = None
        
        # Обработка результатов
        if hasattr(results[0], 'obb') and results[0].obb is not None:
            obb_data = results[0].obb
            if hasattr(obb_data, 'xyxy') and obb_data.xyxy is not None:
                boxes = obb_data.xyxy.cpu().numpy()
                track_ids = obb_data.id.cpu().numpy() if obb_data.id is not None else None

                if track_ids is not None:
                    for box, track_id in zip(boxes, track_ids):
                        x1, y1, x2, y2 = map(int, box)
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2

                        direction_counter.update_counters(track_id, center_x, center_y)
                        
                        if annotated_frame is not None:
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(annotated_frame, f"ID: {track_id}", (x1, y1-10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Сохранение кадра если нужно
        if save_video and annotated_frame is not None:
            out.write(annotated_frame)
            
        # Показ кадра если нужно
        if show_video and annotated_frame is not None:
            cv2.imshow('Car Tracking', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        processed_frames += 1
        pbar.update(1)

    # Закрываем progress bar
    pbar.close()
    cap.release()
    if out is not None:
        out.release()
    if show_video:
        cv2.destroyAllWindows()
    
    # Расчет времени обработки
    processing_time = time.time() - start_time
    
    # Сохранение результатов в CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(['Direction', 'In', 'Out'])
        for direction, counts in direction_counter.directions.items():
            writer.writerow([direction, counts['in'], counts['out']])

    # Вывод статистики
    print(f"\nВидео {video_name} обработано за {processing_time:.2f} сек")
    print(f"Скорость обработки: {total_frames/processing_time:.2f} кадров/сек")
    print(f"Результаты сохранены в {csv_path}")
    if save_video:
        print(f"Обработанное видео сохранено в {video_output_path}")
    print()
    
    return direction_counter.directions

def main():
    # Настройки (можно изменить)
    VIDEO_DIR = "videos"  # Папка с видео
    MODEL_PATH = 'D://Users//Андрей//Desktop//ncontrol-msi//runs//obb//startup-nano32//weights//last.pt'
    OUTPUT_DIR = "results"  # Папка для результатов
    SHOW_VIDEO = False  # Показывать ли видео в процессе обработки
    SAVE_VIDEO = True   # Сохранять ли обработанные видео
    
    # Инициализация модели
    model = YOLO(MODEL_PATH)
    
    # Получаем список видеофайлов
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(('.mp4', '.avi', '.mov'))]
    
    if not video_files:
        print(f"В папке {VIDEO_DIR} не найдено видеофайлов (поддерживаемые форматы: .mp4, .avi, .mov)")
        return
    
    print(f"Найдено {len(video_files)} видеофайлов для обработки")
    
    # Обрабатываем все видео
    all_results = {}
    for video_file in video_files:
        video_path = os.path.join(VIDEO_DIR, video_file)
        results = process_video(
            video_path, 
            model, 
            output_dir=OUTPUT_DIR, 
            show_video=SHOW_VIDEO, 
            save_video=SAVE_VIDEO
        )
        if results:
            all_results[video_file] = results
    
    # Вывод сводной статистики
    print("\nСводная статистика по всем видео:")
    print("{:<20} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
        'Video', 'West In', 'West Out', 'East In', 'East Out', 
        'North In', 'North Out', 'South In', 'South Out'))
    print("-" * 100)
    
    for video_name, counts in all_results.items():
        print("{:<20} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
            os.path.splitext(video_name)[0],
            counts['west']['in'], counts['west']['out'],
            counts['east']['in'], counts['east']['out'],
            counts['north']['in'], counts['north']['out'],
            counts['south']['in'], counts['south']['out']))

if __name__ == "__main__":
    main()