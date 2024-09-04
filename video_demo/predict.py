import io
from typing import Any

import numpy as np
import torch
from cog import BasePredictor, Input, Path
from decord import cpu, VideoReader, bridge
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "THUDM/cogvlm2-video-llama3-chat"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[
    0] >= 8 else torch.float16


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        local_files_only = False  # set to true if models are cached
        cache_dir = "model_cache"
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            cache_dir=cache_dir,
            trust_remote_code=True,
            # padding_side="left"
        )
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                torch_dtype=TORCH_TYPE,
                trust_remote_code=True,
                cache_dir=cache_dir,
                local_files_only=local_files_only
            ).eval().to(DEVICE)
        )

    def predict(
            self,
            prompt: str = Input(description="Prompt", default="Describe this video in all details"),
            video: Path = Input(description="Video"),
            temperature: float = Input(default=0.7, description="Temperature"),
    ) -> Any:
        strategy = 'chat'

        video = self.load_video(video, strategy=strategy)

        history = []
        query = prompt
        inputs = self.model.build_conversation_input_ids(
            tokenizer=self.tokenizer,
            query=query,
            images=[video],
            history=history,
            template_version=strategy
        )
        inputs = {
            'input_ids': inputs['input_ids'].unsqueeze(0).to(DEVICE),
            'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(DEVICE),
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to(DEVICE),
            'images': [[inputs['images'][0].to(DEVICE).to(TORCH_TYPE)]],
        }
        gen_kwargs = {
            "max_new_tokens": 2048,
            "pad_token_id": 128002,
            "top_k": 1,
            "do_sample": False,
            "top_p": 0.1,
            "temperature": temperature,
        }
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    @staticmethod
    def load_video(video: Path, strategy='chat') -> Any:
        bridge.set_bridge('torch')
        with open(video, 'rb') as f:
            mp4_stream = f.read()
        num_frames = 24
        decord_vr = VideoReader(io.BytesIO(mp4_stream), ctx=cpu(0))
        frame_id_list = None
        total_frames = len(decord_vr)
        if strategy == 'base':
            clip_end_sec = 60
            clip_start_sec = 0
            start_frame = int(clip_start_sec * decord_vr.get_avg_fps())
            end_frame = min(total_frames, int(clip_end_sec * decord_vr.get_avg_fps())) if clip_end_sec is not None else total_frames
            frame_id_list = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
        elif strategy == 'chat':
            timestamps = decord_vr.get_frame_timestamp(np.arange(total_frames))
            timestamps = [i[0] for i in timestamps]
            max_second = round(max(timestamps)) + 1
            frame_id_list = []
            for second in range(max_second):
                closest_num = min(timestamps, key=lambda x: abs(x - second))
                index = timestamps.index(closest_num)
                frame_id_list.append(index)
                if len(frame_id_list) >= num_frames:
                    break

        video_data = decord_vr.get_batch(frame_id_list)
        video_data = video_data.permute(3, 0, 1, 2)
        return video_data
