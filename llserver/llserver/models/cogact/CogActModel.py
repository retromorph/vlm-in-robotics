import os

import asyncio
from PIL import Image
import uuid
import torch
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
os.environ["HF_HOME"] = "/home/models"
os.environ["HF_TOKEN"] = "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"

import warnings
from vla import load_vla
from llserver.utils.custom_logger import CustomLogger  # Импортируем логгер

warnings.filterwarnings("ignore")

class CogActModel:
    def __init__(self, model_path="CogACT/CogACT-Base", device="cuda:0", logger=None, action_model_type="DiT-B"):
        self.device = device
        self.model_path = model_path
        self.logger = logger or CustomLogger(mode="logging")
        self.action_model_type = action_model_type
        
        # Загрузка модели CogACT
        self.model = load_vla(
            self.model_path,
            load_for_training=False,
            action_model_type=self.action_model_type,
            future_action_window_size=15,
        )
        
        # Перемещаем модель на указанное устройство и переводим в режим оценки
        self.model.to(self.device).eval()
        
        # Опционально: можно использовать bfloat16 для экономии памяти
        # self.model.vlm = self.model.vlm.to(torch.bfloat16)

        self.task_queue = asyncio.Queue()  # Очередь задач
        self.result_queue = {}  # Результаты задач
        self.task_status = {}  # Статусы задач

    async def _process_task(self, task_id, image_paths, prompt):
        """
        Асинхронная обработка одной задачи с несколькими изображениями.
        """
        self.task_status[task_id] = "in progress"
        self.logger.log(f"Задача с ID {task_id} начала выполняться")

        # Загрузка изображений
        if len(image_paths) > 1:
            self.logger.log("Warning: The model can only process one image at a time. Only the first image will be used.")
        image = Image.open(image_paths[0])

        # Генерация ответа модели CogACT
        actions, generated_text = self.model.predict_action(
            image,
            prompt,
            unnorm_key='fractal20220817_data',  # ключ для нормализации данных
            cfg_scale=1.5,                      # параметр CFG
            use_ddim=True,                      # использование DDIM сэмплирования
            num_ddim_steps=10,                  # количество шагов DDIM
        )

        # Обновляем результат и статус задачи
        self.result_queue[task_id] = {"action": actions.tolist(), "text": generated_text}
        torch.cuda.empty_cache()
        self.task_status[task_id] = "completed"
        self.logger.log(f"Задача {task_id} успешно завершена")

    async def _task_worker(self):
        """
        Асинхронный воркер, который запускает обработку задач параллельно.
        """
        while True:
            if not self.task_queue.empty():
                task_id, image_paths, prompt = await self.task_queue.get()
                self.task_status[task_id] = "in queue"
                asyncio.create_task(self._process_task(task_id, image_paths, prompt))
            await asyncio.sleep(0.1)

    async def put_task(self, image_paths, prompt):
        """
        Добавляет задачу с несколькими изображениями в очередь.
        """
        task_id = str(uuid.uuid4())
        self.task_status[task_id] = "in queue"
        await self.task_queue.put((task_id, image_paths, prompt))
        self.logger.log(f"Задача с ID {task_id} добавлена в очередь с {len(image_paths)} изображениями")
        return task_id

    async def get_task_result(self, task_id):
        """
        Получает результат выполнения задачи по её ID.
        """
        if task_id in self.result_queue:
            return {"status": "completed", "result": self.result_queue.pop(task_id)}
        elif task_id in self.task_status:
            status = self.task_status[task_id]
            if status == "in queue":
                return {"status": "in queue"}
            elif status == "in progress":
                return {"status": "in progress"}
        return {"status": "not found"}
