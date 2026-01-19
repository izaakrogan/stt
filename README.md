# STT Benchmark

Some benchmarking scripts where we're compaing Word Error Rate across different speech-to-text models

## Datasets

- **English** (`en`): LibriSpeech ASR
- **Portuguese** (`pt_br`): Multilingual LibriSpeech

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

# brazilian portuguese
python benchmark.py --dataset pt_br --sample-size 50

# specific models
python benchmark.py --models groq-whisper deepgram-nova3

# save sample audio clips to listen to
python benchmark.py --save-samples 5 --dataset en
python benchmark.py --save-samples 5 --dataset pt_br
```

Audio samples saved to `samples/` folder. Results saved to `results/` folder.
