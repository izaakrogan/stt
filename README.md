# STT Benchmark

Benchmarking Word Error Rate across speech-to-text models for LiveKit voice agent selection.

## Results

| Model               | Portuguese (WER %) | Avg Inference Time (s) | Samples |
|---------------------|--------------------|------------------------|---------|
| faster-whisper-small| 11.99              | 3.9732                 | 50      |
| deepgram-nova3      | 8.52               | 1.1831                 | 50      |
| assemblyai          | 5.79               | 4.9879                 | 50      |
| groq-whisper        | 5.35               | 2.5768                 | 50      |

> **Note:** Run this again with more samples; I did it quick! 100 or more is better.

> **Note:** For voice agents, streaming STT matters more than batch speed. Deepgram and AssemblyAI support streaming (partial transcripts as the user speaks). Groq does not; so despite good accuracy, it's not good for our use case.

## Other models worth testing

- Google Cloud Speech-to-Text
- Azure Speech
- Gladia
- faster-whisper-large-v3 (via Speaches for streaming)

## Datasets

- **English** (`en`): LibriSpeech ASR
- **Portuguese** (`pt`): Multilingual LibriSpeech

## Setup
```bash
pip install -r requirements.txt
```

Create a `.env` file:
```
DEEPGRAM_API_KEY=xxx
ASSEMBLYAI_API_KEY=xxx
GROQ_API_KEY=xxx
```

## Usage
```bash
# run default models on english
python benchmark.py --sample-size 100

# run all models
python benchmark.py --all-models --sample-size 100

# portuguese
python benchmark.py --dataset pt --sample-size 50

# specific models
python benchmark.py --models groq-whisper deepgram-nova3

# save sample audio clips to listen to
python benchmark.py --save-samples 5 --dataset en
python benchmark.py --save-samples 5 --dataset pt
```

Audio samples saved to `samples/` folder. Results saved to `results/` folder.