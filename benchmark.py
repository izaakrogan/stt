#!/usr/bin/env python3
"""STT benchmark - compares WER across different models"""

import argparse
import csv
import io
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import soundfile as sf
import torch
from datasets import load_dataset, Audio
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

from utils import calculate_wer


@dataclass
class BenchmarkResult:
    reference: str
    hypothesis: str
    wer: float
    inference_time: float


@dataclass
class ModelResult:
    model_name: str
    dataset_name: str
    avg_wer: float
    avg_inference_time: float
    total_samples: int


MODEL_REGISTRY: dict[str, Callable] = {}


def register_model(name: str):
    def decorator(func: Callable):
        MODEL_REGISTRY[name] = func
        return func
    return decorator


# faster-whisper (best self-hosted option)
@register_model("faster-whisper-small")
def load_faster_whisper_small():
    from faster_whisper import WhisperModel as FWModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    return FasterWhisperModel(FWModel("small", device=device, compute_type=compute_type), "faster-whisper-small")


# cloud APIs
@register_model("deepgram-nova3")
def load_deepgram_nova3():
    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        raise ValueError("DEEPGRAM_API_KEY not set")
    return DeepgramModel(api_key, "deepgram-nova3")


@register_model("assemblyai")
def load_assemblyai():
    api_key = os.environ.get("ASSEMBLYAI_API_KEY")
    if not api_key:
        raise ValueError("ASSEMBLYAI_API_KEY not set")
    return AssemblyAIModel(api_key, "assemblyai")


@register_model("groq-whisper")
def load_groq_whisper():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")
    return GroqWhisperModel(api_key, "groq-whisper")


def audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """convert audio array to wav bytes for uploading"""
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    buffer.seek(0)
    return buffer.read()


class BaseModel:
    def __init__(self, name: str):
        self.name = name

    def transcribe(self, audio: np.ndarray, sample_rate: int, language: str) -> str:
        raise NotImplementedError


class FasterWhisperModel(BaseModel):
    def __init__(self, model, name: str):
        super().__init__(name)
        self.model = model

    def transcribe(self, audio: np.ndarray, sample_rate: int, language: str) -> str:
        audio = audio.astype(np.float32)
        segments, _ = self.model.transcribe(audio, language=language)
        return " ".join(segment.text for segment in segments)


class DeepgramModel(BaseModel):
    def __init__(self, api_key: str, name: str):
        super().__init__(name)
        self.api_key = api_key

    def transcribe(self, audio: np.ndarray, sample_rate: int, language: str) -> str:
        import httpx

        wav_bytes = audio_to_wav_bytes(audio, sample_rate)
        lang_map = {"en": "en", "pt": "pt-BR"}
        lang = lang_map.get(language, language)

        response = httpx.post(
            "https://api.deepgram.com/v1/listen",
            params={"model": "nova-3", "language": lang},
            headers={"Authorization": f"Token {self.api_key}", "Content-Type": "audio/wav"},
            content=wav_bytes,
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()["results"]["channels"][0]["alternatives"][0]["transcript"]


class AssemblyAIModel(BaseModel):
    def __init__(self, api_key: str, name: str):
        super().__init__(name)
        self.api_key = api_key

    def _request_with_retry(self, method, url, headers, **kwargs):
        import httpx

        max_retries = 5
        for attempt in range(max_retries):
            try:
                if method == "get":
                    response = httpx.get(url, headers=headers, timeout=30.0, **kwargs)
                else:
                    response = httpx.post(url, headers=headers, timeout=30.0, **kwargs)
                response.raise_for_status()
                return response
            except (httpx.ConnectError, httpx.ReadTimeout) as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)

    def transcribe(self, audio: np.ndarray, sample_rate: int, language: str) -> str:
        wav_bytes = audio_to_wav_bytes(audio, sample_rate)
        headers = {"Authorization": self.api_key}

        upload_response = self._request_with_retry("post", "https://api.assemblyai.com/v2/upload", headers, content=wav_bytes)
        audio_url = upload_response.json()["upload_url"]

        lang_map = {"en": "en", "pt": "pt"}
        lang = lang_map.get(language, language)

        transcript_response = self._request_with_retry(
            "post",
            "https://api.assemblyai.com/v2/transcript",
            headers,
            json={"audio_url": audio_url, "language_code": lang}
        )
        transcript_id = transcript_response.json()["id"]

        polling_url = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"
        while True:
            poll_response = self._request_with_retry("get", polling_url, headers)
            status = poll_response.json()

            if status["status"] == "completed":
                return status["text"] or ""
            elif status["status"] == "error":
                raise RuntimeError(f"AssemblyAI error: {status.get('error')}")

            time.sleep(0.5)


class GroqWhisperModel(BaseModel):
    def __init__(self, api_key: str, name: str):
        super().__init__(name)
        self.api_key = api_key

    def transcribe(self, audio: np.ndarray, sample_rate: int, language: str) -> str:
        import httpx

        wav_bytes = audio_to_wav_bytes(audio, sample_rate)

        # retry with backoff for rate limits
        max_retries = 5
        for attempt in range(max_retries):
            response = httpx.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                files={"file": ("audio.wav", wav_bytes, "audio/wav")},
                data={"model": "whisper-large-v3", "language": language},
                timeout=30.0,
            )
            if response.status_code == 429:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
                continue
            response.raise_for_status()
            return response.json()["text"]

        raise RuntimeError("Groq rate limit exceeded after retries")


def load_dataset_samples(dataset_name: str, sample_size: int, seed: int = 42) -> Iterator[tuple[np.ndarray, int, str]]:
    """loads samples from huggingface datasets, streams to avoid memory isues"""

    if dataset_name == "en" or dataset_name == "librispeech":
        ds = load_dataset("librispeech_asr", "clean", split="test", streaming=True)
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        text_key = "text"

    elif dataset_name == "pt":
        ds = load_dataset("facebook/multilingual_librispeech", "portuguese", split="test", streaming=True)
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        text_key = "transcript"

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    ds = ds.shuffle(seed=seed)

    count = 0
    for sample in ds:
        if count >= sample_size:
            break

        audio_data = sample["audio"]
        reference = sample[text_key]

        if not reference or not reference.strip():
            continue

        yield audio_data["array"], audio_data["sampling_rate"], reference
        count += 1


def benchmark_model(model: BaseModel, dataset_name: str, sample_size: int, language: str) -> ModelResult:
    """runs the benchmark for a single model"""
    results: list[BenchmarkResult] = []
    samples = load_dataset_samples(dataset_name, sample_size)

    for audio, sample_rate, reference in tqdm(samples, total=sample_size, desc=f"  {model.name}"):
        start_time = time.perf_counter()
        hypothesis = model.transcribe(audio, sample_rate, language)
        inference_time = time.perf_counter() - start_time

        wer_score = calculate_wer(reference, hypothesis, language)
        results.append(BenchmarkResult(reference=reference, hypothesis=hypothesis, wer=wer_score, inference_time=inference_time))

    avg_wer = np.mean([r.wer for r in results])
    avg_time = np.mean([r.inference_time for r in results])

    return ModelResult(
        model_name=model.name,
        dataset_name=dataset_name,
        avg_wer=avg_wer,
        avg_inference_time=avg_time,
        total_samples=len(results),
    )


def print_results_table(results: list[ModelResult]):
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Model':<25} {'Dataset':<10} {'WER (%)':<10} {'Avg Time':<12} {'Samples':<8}")
    print("-" * 80)

    for r in sorted(results, key=lambda r: (r.dataset_name, r.avg_wer)):
        print(f"{r.model_name:<25} {r.dataset_name:<10} {r.avg_wer * 100:>8.2f}% {r.avg_inference_time:>10.3f}s {r.total_samples:>6}")

    print("=" * 80)


def save_results_csv(results: list[ModelResult], output_path: Path):
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Dataset", "WER (%)", "Avg Inference Time (s)", "Samples"])
        for r in results:
            writer.writerow([r.model_name, r.dataset_name, f"{r.avg_wer * 100:.2f}", f"{r.avg_inference_time:.4f}", r.total_samples])

    print(f"\nResults saved to: {output_path}")


def save_audio_samples(dataset_name: str, num_samples: int, seed: int = 42):
    """saves some sample audio clips so you can listen to them"""
    samples_dir = Path("samples") / dataset_name
    samples_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving {num_samples} audio samples to {samples_dir}/...")
    samples = load_dataset_samples(dataset_name, num_samples, seed)

    for i, (audio, sample_rate, reference) in enumerate(samples):
        audio_path = samples_dir / f"sample_{i+1:03d}.wav"
        sf.write(audio_path, audio, sample_rate)

        txt_path = samples_dir / f"sample_{i+1:03d}.txt"
        txt_path.write_text(reference)

        print(f"  Saved: {audio_path.name} - \"{reference[:50]}{'...' if len(reference) > 50 else ''}\"")

    print(f"\nSamples saved to: {samples_dir}/")


def main():
    parser = argparse.ArgumentParser(description="STT benchmark")

    parser.add_argument("--sample-size", type=int, default=100, help="number of utterances to test")
    parser.add_argument("--models", nargs="+", choices=list(MODEL_REGISTRY.keys()), help="which models to run")
    parser.add_argument("--all-models", action="store_true", help="run all availible models")
    parser.add_argument("--dataset", choices=["en", "pt", "all"], default="pt")
    parser.add_argument("--output", type=Path, default=None, help="output csv path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-samples", type=int, default=0, metavar="N", help="save N audio samples to listen to")

    args = parser.parse_args()

    # setup results folder
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = results_dir / f"benchmark_{timestamp}.csv"

    # figure out which models to run
    if args.all_models:
        model_names = list(MODEL_REGISTRY.keys())
    elif args.models:
        model_names = args.models
    else:
        model_names = ["faster-whisper-small", "deepgram-nova3", "assemblyai", "groq-whisper"]

    datasets = ["en", "pt"] if args.dataset == "all" else [args.dataset]

    # save audio samples if reqested
    if args.save_samples > 0:
        for dataset_name in datasets:
            save_audio_samples(dataset_name, args.save_samples, args.seed)
        print()

    device = "CUDA" if torch.cuda.is_available() else "CPU"
    print(f"\nRunning benchmark on: {device}")
    print(f"Sample size: {args.sample_size}")
    print(f"Models: {', '.join(model_names)}")
    print(f"Datasets: {', '.join(datasets)}")
    print()

    all_results: list[ModelResult] = []

    for dataset_name in datasets:
        print(f"\n--- Dataset: {dataset_name} ---")

        for model_name in model_names:
            print(f"\nLoading {model_name}...")
            try:
                model = MODEL_REGISTRY[model_name]()
            except Exception as e:
                print(f"  Failed to load {model_name}: {e}")
                continue

            result = benchmark_model(model, dataset_name, args.sample_size, dataset_name)
            all_results.append(result)

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if all_results:
        print_results_table(all_results)
        save_results_csv(all_results, args.output)
    else:
        print("\nNo results to display")


if __name__ == "__main__":
    main()
