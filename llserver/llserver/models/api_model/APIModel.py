import asyncio
import uuid
import warnings

warnings.filterwarnings("ignore")

from openai import OpenAI
import base64
import hashlib


# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


# Function to get string and calculate its hash using MD5
def get_string_hash(input_string):
    """
    Function to calculate the MD5 hash of a given string.
    """
    return hashlib.md5(input_string.encode()).hexdigest()


class APIModel:
    def __init__(self, logger=None):
        self.logger = logger
        self.logger.log(f"Начинаем приветствие...")
        # https://aitunnel.ru/#models
        self.client = OpenAI(
            api_key="sk-aitunnel-lRVzfpdyQrVrYJGyxiFkgMMXLSz5P2Oz", # Ключ из нашего сервиса
            base_url="https://api.aitunnel.ru/v1/",
        )
        
        self.task_queue = asyncio.Queue()  # Очередь задач
        self.result_queue = {}  # Результаты задач
        self.task_status = {}  # Статусы задач
        self.logger.log(f"Привет, я ApiModel")


    def _predict(self, prompt, image_paths=[], model="gemini-pro-1.5", **kwargs):
        content = [
            {
                "type": "text",
                "text": prompt
            }
        ]
        
        for image_path in image_paths:
            base64_image = encode_image(image_path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })
        
        messages=[
            {
                "role": "user",
                "content": content
            }
        ]
        # Log the prediction details
        prompt_hash = get_string_hash(prompt)
        self.logger.log(f"Predicting with prompt (hash={prompt_hash}), images number={len(image_paths)} and model name={model}")
        self.logger.log(f"kwargs: {kwargs}")
        
        completion = self.client.chat.completions.create(
            messages=messages,
            max_tokens=1000,
            model=model
        )

        text_outputs = completion.choices[0].message.content
        return text_outputs

    async def _process_task(self, task_id, image_paths, prompt, kwargs):
        """
        Асинхронная обработка одной задачи с несколькими изображениями.
        """        
        # Обновляем статус задачи до "in progress"
        self.task_status[task_id] = "in progress"
        self.logger.log(f"Задача с ID {task_id} начала выполняться")

        self.logger.log(f"kwargs: {kwargs}")
        response = self._predict(prompt, image_paths, **kwargs)
        self.logger.log(f"response: {response}")
        
        # Обновляем результат и статус задачи
        self.result_queue[task_id] = response

        self.task_status[task_id] = "completed"
        self.logger.log(f"Задача {task_id} успешно завершена")

    async def _task_worker(self):
        """
        Асинхронный воркер, который запускает обработку задач параллельно.
        """
        while True:
            if not self.task_queue.empty():
                task_id, image_path, prompt, kwargs = await self.task_queue.get()
                self.task_status[task_id] = "in queue"  # Статус задачи "в очереди"
                self.logger.log(f"kwargs: {kwargs}")
                asyncio.create_task(self._process_task(task_id, image_path, prompt, kwargs))  # Запуск задачи параллельно
            await asyncio.sleep(0.1)  # Задержка для предотвращения блокировки

    async def put_task(self, image_paths, prompt, **kwargs):
        """
        Добавляет задачу с несколькими изображениями в очередь.
        """
        task_id = str(uuid.uuid4())  # Уникальный ID для задачи
        self.task_status[task_id] = "in queue"  # Задача в очереди
        self.logger.log(f"kwargs: {kwargs}")
        await self.task_queue.put((task_id, image_paths, prompt, kwargs))
        self.logger.log(f"Задача с ID {task_id} добавлена в очередь с {len(image_paths)} изображениями")
        return task_id

    async def get_task_result(self, task_id):
        """
        Получает результат выполнения задачи по её ID.
        """
        if task_id in self.result_queue:
            return {"status": "completed", "result": self.result_queue.pop(task_id)}
        elif task_id in self.task_status:
            # Проверяем статус задачи и возвращаем его
            status = self.task_status[task_id]
            if status == "in queue":
                return {"status": "in queue"}
            elif status == "in progress":
                return {"status": "in progress"}
        # Если задача не найдена
        return {"status": "not found"}