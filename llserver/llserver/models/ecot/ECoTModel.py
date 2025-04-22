import os

import asyncio
from PIL import Image
import uuid
import torch
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
os.environ["HF_HOME"] = "/home/models"

import warnings
from transformers import AutoProcessor, AutoModelForVision2Seq
from llserver.utils.custom_logger import CustomLogger  # Импортируем логгер

warnings.filterwarnings("ignore")

class ECoTModel:
    def __init__(self, model_path="Embodied-CoT/ecot-openvla-7b-bridge", device="cuda:2", logger=None):
        self.device = device
        self.model_path = model_path
        self.logger = logger or CustomLogger(mode="logging")
        
        # Загрузка модели и процессора
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(self.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).to(self.device)
        self.model.eval()

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
        inputs = self.processor(prompt, image).to(self.device, dtype=torch.bfloat16)

        # Генерация ответа модели
        action, generated_ids = self.model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False, max_new_tokens=1024)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        # Обновляем результат и статус задачи
        self.result_queue[task_id] = {"action": action.tolist(), "text": generated_text}
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
