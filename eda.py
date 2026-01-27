#!/usr/bin/env python3
"""Exploratory Data Analysis for STT benchmark datasets"""

import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset, Audio
from collections import Counter
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

SAMPLE_SIZE = 500  # Analyze 500 samples per dataset

def load_samples(dataset_name: str, sample_size: int, seed: int = 42):
    """Load samples and extract metadata"""
    print(f"Loading {dataset_name} dataset...")

    if dataset_name == "en":
        ds = load_dataset("openslr/librispeech_asr", "clean", split="test", streaming=True, trust_remote_code=True)
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        text_key = "text"
        speaker_key = "speaker_id"
    elif dataset_name == "pt":
        ds = load_dataset("facebook/multilingual_librispeech", "portuguese", split="test", streaming=True, trust_remote_code=True)
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        text_key = "transcript"
        speaker_key = "speaker_id"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    ds = ds.shuffle(seed=seed)

    samples = []
    for i, sample in enumerate(ds):
        if i >= sample_size:
            break

        audio = sample["audio"]["array"]
        sr = sample["audio"]["sampling_rate"]
        text = sample[text_key]

        if not text or not text.strip():
            continue

        duration = len(audio) / sr
        word_count = len(text.split())
        char_count = len(text)
        speaker = sample.get(speaker_key, "unknown")

        samples.append({
            "duration": duration,
            "word_count": word_count,
            "char_count": char_count,
            "words_per_second": word_count / duration if duration > 0 else 0,
            "speaker": speaker,
            "text": text,
            "sample_rate": sr,
        })

    print(f"  Loaded {len(samples)} samples")
    return samples


def plot_duration_distribution(en_samples, pt_samples):
    """Plot audio duration distributions"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    en_durations = [s["duration"] for s in en_samples]
    pt_durations = [s["duration"] for s in pt_samples]

    # Histograms
    axes[0].hist(en_durations, bins=30, alpha=0.7, label="English (LibriSpeech)", color="#2ecc71", edgecolor="black")
    axes[0].hist(pt_durations, bins=30, alpha=0.7, label="Portuguese (MLS)", color="#3498db", edgecolor="black")
    axes[0].set_xlabel("Duration (seconds)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Audio Duration Distribution")
    axes[0].legend()
    axes[0].axvline(np.mean(en_durations), color="#27ae60", linestyle="--", label=f"EN mean: {np.mean(en_durations):.1f}s")
    axes[0].axvline(np.mean(pt_durations), color="#2980b9", linestyle="--", label=f"PT mean: {np.mean(pt_durations):.1f}s")

    # Box plots
    axes[1].boxplot([en_durations, pt_durations], labels=["English", "Portuguese"])
    axes[1].set_ylabel("Duration (seconds)")
    axes[1].set_title("Duration Comparison (Box Plot)")

    plt.tight_layout()
    plt.savefig("eda_duration.png", dpi=150)
    print("Saved: eda_duration.png")
    plt.close()


def plot_text_statistics(en_samples, pt_samples):
    """Plot word count and character distributions"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    en_words = [s["word_count"] for s in en_samples]
    pt_words = [s["word_count"] for s in pt_samples]
    en_chars = [s["char_count"] for s in en_samples]
    pt_chars = [s["char_count"] for s in pt_samples]

    # Word count histogram
    axes[0, 0].hist(en_words, bins=25, alpha=0.7, label="English", color="#2ecc71", edgecolor="black")
    axes[0, 0].hist(pt_words, bins=25, alpha=0.7, label="Portuguese", color="#3498db", edgecolor="black")
    axes[0, 0].set_xlabel("Word Count")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Words per Utterance")
    axes[0, 0].legend()

    # Character count histogram
    axes[0, 1].hist(en_chars, bins=25, alpha=0.7, label="English", color="#2ecc71", edgecolor="black")
    axes[0, 1].hist(pt_chars, bins=25, alpha=0.7, label="Portuguese", color="#3498db", edgecolor="black")
    axes[0, 1].set_xlabel("Character Count")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Characters per Utterance")
    axes[0, 1].legend()

    # Words per second (speech rate)
    en_wps = [s["words_per_second"] for s in en_samples]
    pt_wps = [s["words_per_second"] for s in pt_samples]

    axes[1, 0].hist(en_wps, bins=25, alpha=0.7, label="English", color="#2ecc71", edgecolor="black")
    axes[1, 0].hist(pt_wps, bins=25, alpha=0.7, label="Portuguese", color="#3498db", edgecolor="black")
    axes[1, 0].set_xlabel("Words per Second")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_title("Speech Rate Distribution")
    axes[1, 0].legend()

    # Scatter: duration vs word count
    axes[1, 1].scatter([s["duration"] for s in en_samples], en_words, alpha=0.5, label="English", color="#2ecc71", s=20)
    axes[1, 1].scatter([s["duration"] for s in pt_samples], pt_words, alpha=0.5, label="Portuguese", color="#3498db", s=20)
    axes[1, 1].set_xlabel("Duration (seconds)")
    axes[1, 1].set_ylabel("Word Count")
    axes[1, 1].set_title("Duration vs Word Count")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig("eda_text_stats.png", dpi=150)
    print("Saved: eda_text_stats.png")
    plt.close()


def plot_speaker_distribution(en_samples, pt_samples):
    """Plot speaker distribution"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    en_speakers = Counter([s["speaker"] for s in en_samples])
    pt_speakers = Counter([s["speaker"] for s in pt_samples])

    # Number of samples per speaker (top 20)
    en_top = en_speakers.most_common(20)
    pt_top = pt_speakers.most_common(20)

    if en_top:
        axes[0].bar(range(len(en_top)), [x[1] for x in en_top], color="#2ecc71", edgecolor="black")
        axes[0].set_xlabel("Speaker ID (top 20)")
        axes[0].set_ylabel("Sample Count")
        axes[0].set_title(f"English: {len(en_speakers)} unique speakers")
        axes[0].set_xticks(range(len(en_top)))
        axes[0].set_xticklabels([str(x[0])[:6] for x in en_top], rotation=45, ha="right")

    if pt_top:
        axes[1].bar(range(len(pt_top)), [x[1] for x in pt_top], color="#3498db", edgecolor="black")
        axes[1].set_xlabel("Speaker ID (top 20)")
        axes[1].set_ylabel("Sample Count")
        axes[1].set_title(f"Portuguese: {len(pt_speakers)} unique speakers")
        axes[1].set_xticks(range(len(pt_top)))
        axes[1].set_xticklabels([str(x[0])[:6] for x in pt_top], rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig("eda_speakers.png", dpi=150)
    print("Saved: eda_speakers.png")
    plt.close()


def plot_summary_statistics(en_samples, pt_samples):
    """Create summary statistics table as figure"""

    def calc_stats(samples, name):
        durations = [s["duration"] for s in samples]
        words = [s["word_count"] for s in samples]
        wps = [s["words_per_second"] for s in samples]
        speakers = len(set(s["speaker"] for s in samples))

        return {
            "Dataset": name,
            "Samples": len(samples),
            "Speakers": speakers,
            "Duration (mean)": f"{np.mean(durations):.2f}s",
            "Duration (std)": f"{np.std(durations):.2f}s",
            "Duration (min)": f"{np.min(durations):.2f}s",
            "Duration (max)": f"{np.max(durations):.2f}s",
            "Words (mean)": f"{np.mean(words):.1f}",
            "Words (std)": f"{np.std(words):.1f}",
            "Speech rate (words/s)": f"{np.mean(wps):.2f}",
        }

    en_stats = calc_stats(en_samples, "English (LibriSpeech)")
    pt_stats = calc_stats(pt_samples, "Portuguese (MLS)")

    # Print table
    print("\n" + "=" * 60)
    print("DATASET SUMMARY STATISTICS")
    print("=" * 60)
    print(f"{'Metric':<25} {'English':<15} {'Portuguese':<15}")
    print("-" * 60)

    for key in en_stats:
        if key == "Dataset":
            continue
        print(f"{key:<25} {en_stats[key]:<15} {pt_stats[key]:<15}")
    print("=" * 60)

    # Create figure with table
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')

    table_data = []
    headers = ["Metric", "English (LibriSpeech)", "Portuguese (MLS)"]
    for key in en_stats:
        if key == "Dataset":
            continue
        table_data.append([key, en_stats[key], pt_stats[key]])

    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        loc='center',
        cellLoc='center',
        colColours=['#f0f0f0'] * 3,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    plt.title("Dataset Summary Statistics", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig("eda_summary.png", dpi=150, bbox_inches='tight')
    print("Saved: eda_summary.png")
    plt.close()


def plot_correlation_heatmap(en_samples, pt_samples):
    """Plot correlation between numeric features"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, samples, title in [(axes[0], en_samples, "English"), (axes[1], pt_samples, "Portuguese")]:
        data = {
            "duration": [s["duration"] for s in samples],
            "words": [s["word_count"] for s in samples],
            "chars": [s["char_count"] for s in samples],
            "wps": [s["words_per_second"] for s in samples],
        }

        corr_matrix = np.corrcoef([data["duration"], data["words"], data["chars"], data["wps"]])

        im = ax.imshow(corr_matrix, cmap="RdYlGn", vmin=-1, vmax=1)
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(["Duration", "Words", "Chars", "WPS"], rotation=45, ha="right")
        ax.set_yticklabels(["Duration", "Words", "Chars", "WPS"])
        ax.set_title(f"{title} Feature Correlations")

        # Add correlation values
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{corr_matrix[i, j]:.2f}", ha="center", va="center", fontsize=10)

    plt.colorbar(im, ax=axes, shrink=0.8, label="Correlation")
    plt.tight_layout()
    plt.savefig("eda_correlation.png", dpi=150)
    print("Saved: eda_correlation.png")
    plt.close()


def plot_cumulative_duration(en_samples, pt_samples):
    """Plot cumulative duration distribution"""
    fig, ax = plt.subplots(figsize=(10, 6))

    en_durations = sorted([s["duration"] for s in en_samples])
    pt_durations = sorted([s["duration"] for s in pt_samples])

    ax.plot(en_durations, np.linspace(0, 1, len(en_durations)), label="English", color="#2ecc71", linewidth=2)
    ax.plot(pt_durations, np.linspace(0, 1, len(pt_durations)), label="Portuguese", color="#3498db", linewidth=2)

    ax.set_xlabel("Duration (seconds)")
    ax.set_ylabel("Cumulative Proportion")
    ax.set_title("Cumulative Distribution of Audio Durations")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add percentile markers
    for pct in [0.25, 0.5, 0.75]:
        ax.axhline(y=pct, color='gray', linestyle='--', alpha=0.5)
        ax.text(max(en_durations + pt_durations) * 0.95, pct + 0.02, f"{int(pct*100)}%", fontsize=9)

    plt.tight_layout()
    plt.savefig("eda_cumulative.png", dpi=150)
    print("Saved: eda_cumulative.png")
    plt.close()


def main():
    print("=" * 60)
    print("STT BENCHMARK - EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    print(f"\nAnalyzing {SAMPLE_SIZE} samples from each dataset...\n")

    # Load data
    en_samples = load_samples("en", SAMPLE_SIZE)
    pt_samples = load_samples("pt", SAMPLE_SIZE)

    print("\nGenerating visualizations...\n")

    # Generate all plots
    plot_summary_statistics(en_samples, pt_samples)
    plot_duration_distribution(en_samples, pt_samples)
    plot_text_statistics(en_samples, pt_samples)
    plot_speaker_distribution(en_samples, pt_samples)
    plot_correlation_heatmap(en_samples, pt_samples)
    plot_cumulative_duration(en_samples, pt_samples)

    print("\n" + "=" * 60)
    print("EDA COMPLETE - Generated 6 visualization files:")
    print("  - eda_summary.png")
    print("  - eda_duration.png")
    print("  - eda_text_stats.png")
    print("  - eda_speakers.png")
    print("  - eda_correlation.png")
    print("  - eda_cumulative.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
