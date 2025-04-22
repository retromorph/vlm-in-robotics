import os

import asyncio
import uuid
from PIL import Image
import requests
import copy
import torch
import warnings
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llserver.models.lera_prompts import PROMPTS

warnings.filterwarnings("ignore")

MODEL_NAMES = {
    # interleave
    "interleave-0.5b": "lmms-lab/llava-next-interleave-qwen-0.5b",
    "interleave-7b": "lmms-lab/llava-next-interleave-qwen-7b",
    "interleave-7b-dpo": "lmms-lab/llava-next-interleave-qwen-7b-dpo",
    
    # onevision
    "onevision-0.5b": "lmms-lab/llava-onevision-qwen2-0.5b-si",
    "onevision-7b": "lmms-lab/llava-onevision-qwen2-7b-si",
    "onevision-7b-chat": "lmms-lab/llava-onevision-qwen2-7b-ov-chat",
}

class LLERa:
    def __init__(self, pretrained=MODEL_NAMES["onevision-7b-chat"], model_name="llava_qwen", device="cuda", logger=None):
        self.device = device
        self.model_name = model_name
        self.pretrained = pretrained
        self.device_map = "auto"
        self.logger = logger
        self.llava_model_args = {
            "multimodal": True,
            "attn_implementation": "sdpa",
        }

        # Загрузка модели
        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(
            self.pretrained, None, self.model_name, device_map=self.device_map, **self.llava_model_args
        )
        self.model.eval()

        self.task_queue = asyncio.Queue()  # Очередь задач
        self.result_queue = {}  # Результаты задач
        self.task_status = {}  # Статусы задач


    def _predict(self, prompt, image_paths=[]):
        # Создание промпта без добавления токенов изображений
        conv_template = "qwen_1_5"
        question = prompt  # Токены изображений уже содержатся в промпте
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)

        # Загрузка изображений
        if len(image_paths) > 0:
            images = [Image.open(image_path) for image_path in image_paths]
            image_tensors = process_images(images, self.image_processor, self.model.config)
            image_tensors = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensors]
            image_sizes = [image.size for image in images]
            
            # Генерация ответа модели
            cont = self.model.generate(
                input_ids,
                images=image_tensors,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0,
                max_new_tokens=1000,
            )
        else:
            # Генерация ответа модели
            cont = self.model.generate(
                input_ids,
                do_sample=False,
                temperature=0,
                max_new_tokens=1000,
            )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)

        # Очищаем кэш CUDA после завершения задачи
        torch.cuda.empty_cache()

        return text_outputs

    async def _process_task(self, task_id, image_paths, prompt):
        """
        Асинхронная обработка одной задачи с несколькими изображениями.
        """
        def replace_template_with_info(template, info):
            for name, value in info.items():
                template = template.replace(name, value)
            return template
        
        # Обновляем статус задачи до "in progress"
        self.task_status[task_id] = "in progress"
        self.logger.log(f"Задача с ID {task_id} начала выполняться")

        self.logger.log("="*60)
        # prompt = "[goal]###[current_plan]###[available_actions]"
        goal, success_actions, current_plan, available_actions = prompt.split("###")
        info = {
            "[goal]": goal,
            "[success_actions]": success_actions,
            "[current_plan]": current_plan,
            "[available_actions]": available_actions,
        }


        self.logger.log(f"goal: {goal}")
        self.logger.log(f"success_actions: {success_actions}")
        self.logger.log(f"current_plan: {current_plan}")   
        self.logger.log(f"available_actions: {available_actions}")
        self.logger.log("="*30)

        look_request = replace_template_with_info(PROMPTS["look"], info)
        self.logger.log(f"look_request: \n{look_request}\n")
        look_response = self._predict(look_request, image_paths)[0]
        info["[look_response]"] = look_response
        self.logger.log(f"look_response: \n{look_response}\n")
        self.logger.log("="*30)

        explain_request = replace_template_with_info(PROMPTS["explain"], info)
        self.logger.log(f"explain_request: \n{explain_request}\n")
        explain_response = self._predict(explain_request)[0]
        info["[explain_response]"] = explain_response
        self.logger.log(f"explain_response: \n{explain_response}\n")
        self.logger.log("="*30)

        replan_request = replace_template_with_info(PROMPTS["replan"], info)
        self.logger.log(f"replan_request: \n{replan_request}\n")
        replan_response = self._predict(replan_request)[0]
        info["[replan_response]"] = replan_response
        self.logger.log(f"replan_response: \n{replan_response}\n")
        self.logger.log("="*60)

        # Обновляем результат и статус задачи
        self.result_queue[task_id] = replan_response

        self.task_status[task_id] = "completed"
        self.logger.log(f"Задача {task_id} успешно завершена")

    async def _task_worker(self):
        """
        Асинхронный воркер, который запускает обработку задач параллельно.
        """
        while True:
            if not self.task_queue.empty():
                task_id, image_path, prompt = await self.task_queue.get()
                self.task_status[task_id] = "in queue"  # Статус задачи "в очереди"
                asyncio.create_task(self._process_task(task_id, image_path, prompt))  # Запуск задачи параллельно
            await asyncio.sleep(0.1)  # Задержка для предотвращения блокировки

    async def put_task(self, image_paths, prompt):
        """
        Добавляет задачу с несколькими изображениями в очередь.
        """
        task_id = str(uuid.uuid4())  # Уникальный ID для задачи
        self.task_status[task_id] = "in queue"  # Задача в очереди
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
            # Проверяем статус задачи и возвращаем его
            status = self.task_status[task_id]
            if status == "in queue":
                return {"status": "in queue"}
            elif status == "in progress":
                return {"status": "in progress"}
        # Если задача не найдена
        return {"status": "not found"}