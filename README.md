<p align="center">
  <img width="668" height="201" alt="Image" src="https://github.com/user-attachments/assets/d6d715f7-59cb-4321-ad29-97d04dbd2de5" /> <br>
   <a href="https://github.com/netplexflix/ULDAS/releases"><img alt="GitHub Release" src="https://img.shields.io/github/v/release/netplexflix/ULDAS?style=plastic"></a>
   <a href="https://hub.docker.com/repository/docker/netplexflix/uldas"><img alt="Docker Pulls" src="https://img.shields.io/docker/pulls/netplexflix/uldas?style=plastic"></a>
   <a href="https://discord.gg/VBNUJd7tx3"><img alt="Discord" src="https://img.shields.io/discord/1329439972796928041?style=plastic&label=Discord"></a>
</p>

Do you have Movies or TV shows in your media player for which the audio and/or subtitle tracks are labeled as "undefined" or "unknown"?</br>
ULDAS (Undefined Language Detector for Audio and Subtitles) solves that problem by:

1. Scanning your video files for audio and subtitle tracks with undefined language
2. Extracting audio and subtitle samples
3. Using AI speech recognition to detect the audio language
4. Detecting subtitle language (can also detect if subtitles are [FORCED] and/or [SDH])
5. Updating the file metadata with the correct language codes and flags

The script optionally remuxes non MKV video formats to MKV first.

<img width="628" height="58" alt="Image" src="https://github.com/user-attachments/assets/8c1eca62-50cb-4114-9de2-1244dd0a0714" />

Requires
- [Python >=3.11](https://www.python.org/downloads/)
- [FFmpeg](https://ffmpeg.org/download.html)
- [MKVToolNix](https://mkvtoolnix.download/downloads.html)
- [Tesseract-OCR](https://github.com/tesseract-ocr/tesseract?tab=readme-ov-file#installing-tesseract) for image based subtitles (e.g. PGS)

---

## 📑 Table of Contents

- [🛠️ Installation](#installation)
  - [1️⃣ Download the script](#download-the-script)
  - [2️⃣ Install Dependencies](#install-dependencies)
- [🐳 Docker](#docker)
  - [Quick Start](#quick-start)
  - [Volume Mounts](#volume-mounts)
  - [Environment Variables](#environment-variables)
  - [Unraid](#unraid)
- [⚙️ Configuration](#configuration)
  - [Expert variables](#expert-variables)
  - [Model Size Guide](#model-size-guide)
- [🚀 Usage](#usage)
- [📄 Supported File Formats](#supported-file-formats)
- [⌨️ CLI Reference](#cli-reference)
  - [Utility Commands](#utility-commands)
  - [Logging & Output](#logging--output)
  - [Paths & General](#paths--general)
  - [Voice Activity Detection (VAD)](#voice-activity-detection-vad)
  - [Device & Compute](#device--compute)
  - [Confidence & Reprocessing](#confidence--reprocessing)
  - [Tracking](#tracking)
  - [Subtitle Processing](#subtitle-processing)
  - [Timeouts](#timeouts)
  - [Forced Subtitle Thresholds](#forced-subtitle-thresholds)
  - [CLI ↔ Config File Mapping](#cli--config-file-mapping)
- [🏞️ Example run summary](#example-run-summary)
  - [Audio Processing](#audio-processing)
  - [Subtitle Processing](#subtitle-processing-1)
- [⚠️ Need Help or have Feedback?](#need-help-or-have-feedback)
- [❤️ Support the Project](#support-the-project)

---

<a id="installation"></a>
## 🛠️ Installation

<a id="download-the-script"></a>
### 1️⃣ Download the script
Clone the repository:
```sh
git clone https://github.com/netplexflix/ULDAS.git
cd ULDAS
```

![#c5f015](https://placehold.co/15x15/c5f015/c5f015.png) Or simply download by pressing the green 'Code' button above and then 'Download Zip'.

<a id="install-dependencies"></a>
### 2️⃣ Install Dependencies
- Ensure you have [Python](https://www.python.org/downloads/) installed (`>=3.11` recommended)
- Open a Terminal in the script's directory
>[!TIP]
>Windows Users: <br/>
>Go to the script folder (where ULDAS.py is).</br>
>Right mouse click on an empty space in the folder and click `Open in Windows Terminal`
- Install the required dependencies:
```sh
pip install -r requirements.txt
```

***

<a id="docker"></a>
## 🐳 Docker

ULDAS is available as a Docker image. The image supports both CPU and Nvidia GPUs.

<a id="quick-start"></a>
### Quick Start

For CPU (recommended):
```sh
docker run --rm \
  -v /path/to/config:/app/config \
  -v /path/to/folder1:/folder1 \
  -v /path/to/folder2:/folder2 \
  netplexflix/uldas:latest
```

For Nvidia GPU (advanced):

```sh
docker run --rm --gpus all \
  -v /path/to/config:/app/config \
  -v /path/to/folder1:/folder1 \
  -v /path/to/folder2:/folder2 \
  netplexflix/uldas:latest
```

A compose file is avaliable, [here](./docker-compose.yml), aswell for convienence.
1. Download the `docker-compose.yml` file from this repository. Edit the port and timezone if needed
2. Pull the latest image:
```bash
docker compose pull
```
3. Start the container:
```bash
docker compose up -d
```

<a id="volume-mounts"></a>
### Volume Mounts

| Mount | Description |
| --- | --- |
| `/app/config` | Config file and tracking data (required) |
| `/folder1` | e.g. Your movies library |
| `/folder2` | e.g. Your TV shows library |
| `/tmp/uldas` | optional mount for custom tmp directory |


> [!NOTE]
> Your `config.yml` paths need to match the paths inside your container<br>
> example:<br>
> ```sh
>     volumes:
>       - ./config:/app/config
>       - /media/movies:/folder1
>       - /media/tv:/folder2
> ```
> Extra paths can be added as long as they are also listed in the config.

<a id="environment-variables"></a>
### Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `PUID` | `0` | User ID for file permissions |
| `PGID` | `0` | Group ID for file permissions |
| `NVIDIA_VISIBLE_DEVICES` | `all` | Used to specify specific Nvidia GPUs to use |
| `NVIDIA_DRIVER_CAPABILITIES` | `all` | Used to specify specific Nvidia driver capabilities the container has access to |

<a id="unraid"></a>
### Unraid

ULDAS is available in Community Applications. Search for "ULDAS" or install manually using the template in [unraid/](./unraid/).

***

<a id="configuration"></a>
## ⚙️ Configuration
Rename `config.example.yml` to `config.yml` and change the values where needed:

- **path**: Main Paths for your media.
- **remux_to_mkv**: `true` remuxes non-MKV files so they can be processed too
- **show_details**: `true` will show you more details of what's happening
- **dry_run**: `true` will do a dry run (will show what it would do, without actually altering any files)
- **process_subtitles**: `true` will process undefined subtitle tracks
- **process_external_subtitles**: `true` will process external subtitle files
- **analyze_forced_subtitles**: `true` will analyze whether a subtitle track has "Forced Subtitles" or not
- **detect_sdh_subtitles**: `true` will analyze whether a subtitle track has 'hearing impaired' support. (e.g.: [Dogs barking], [Narrator:],... )

<a id="expert-variables"></a>
### Expert variables

> [!TIP]
>You can create a config file with a few expert variables by using the following command:
>```sh
>python ULDAS.py --create-config
>```

Only Change these if you know what you're doing.
- **vad_filter**: Enables Voice Activity Detection to filter out silence and background noise before language analysis (Default: True)
- **vad_min_speech_duration_ms**: Minimum speech segment length (in milliseconds) to consider as valid speech (Default: 250)
- **vad_max_speech_duration_s**: Maximum continuous speech segment length (in seconds) before splitting (Default: 30)
- **whisper_model**: See Model Size Guide below
- **device**: Hardware acceleration preference (auto, cpu, or cuda). Auto-detects CUDA GPU if available, falls back to CPU (Default: "auto")
- **compute_type**: Precision/performance trade-off (auto, int8, float16, float32). Auto-selects optimal type based on device (Default: "auto")
- **cpu_threads**:Number of CPU threads to use. 0 = automatic detection based on system cores (Default: 0)
- **confidence_threshold**: Minimum confidence level (0.0-1.0) required to accept language detection from audio samples. If sample-based detection falls below this threshold, the entire audio track is analyzed for improved accuracy. Higher values are more conservative but reduce false positives. (Default: 0.9)
- **subtitle_confidence_threshold**: If subtitle detection confidence falls below confidence, the track is skipped
- **reprocess_all** : `true` will reprocess ALL audio tracks, even if they already have a language tag. (Default: `false`)
- **reprocess_all_subtitles**: `true` will reprocess ALL subtitle tracks, even if they already have a language tag. (Default: `false`)
- **operation_timeout_seconds**: 600,  # 10 minutes
- **temp_dir**: Change temporary directory for audio/subtitle extraction. Leave empty to use system default (/tmp)

Forced subtitle detection thresholds.<br>
Density-based:
- **forced_subtitle_low_density_threshold**: Below = likely forced
- **forced_subtitle_high_density_threshold**: Above = likely full

Coverage-based (secondary factor):
- **forced_subtitle_low_coverage_threshold**: Below = likely forced
- **forced_subtitle_high_coverage_threshold**: Above = likely full

Absolute count thresholds:
- **forced_subtitle_min_count_threshold**: Below = likely forced
- **forced_subtitle_max_count_threshold**: Above = likely full

<a id="model-size-guide"></a>
### Model Size Guide

- **tiny:** Fastest, least accurate
- **base:** Good balance
- **small:** More accurate, slower (used during development tests)
- **medium:** Very accurate, much slower
- **large:** Most accurate, very slow

***

<a id="usage"></a>
## 🚀 Usage

Run the script with:
```sh
python ULDAS.py
```

> [!TIP]
> Windows users can create a batch file for quick launching:
> ```batch
> "C:\\Path\\To\\Python\\python.exe" "Path\\To\\Script\\ULDAS.py"
> pause
> ```

> [!NOTE]
> ### Audio Tracks
> A warning will be given at the end of a run for any files that were marked as 'zxx' (no linguistic content).<br>
> While it is perfectly possible for a video file to have no linguistic content (silent movies, old Disney cartoons, etc), these could also indicate AI 'hallucinations'.
> You may want to manually check these files.
>
> ### Subtitle Tracks
> Tracks with confidence below the `subtitle_confidence_threshold` are automatically skipped and shown in the summary.
> For image-based (PGS) subtitles without OCR support, language detection will be skipped.
>
> ### Failed Files
> If a file is marked as failed, it is likely corrupt. Manually remux or replace it.
> Renaming of external subtitle files can fail if the targetted name is already in use by another file, or if the file is set to read-only.

***

<a id="supported-file-formats"></a>
## 📄 Supported File Formats
Always Processed:
- **MKV files:** Primary target format

With `remux_to_mkv: true`
- MP4, AVI, MOV, WMV, FLV, WebM, M4V, M2TS, MTS, TS, VOB
- Note: Original files are deleted after successful conversion

External subtitle File Formats:
- SRT, ASS, SSA, SUB, VTT, IDX

***

<a id="cli-reference"></a>
## ⌨️ CLI Reference

<a id="utility-commands"></a>
### Utility Commands

| Flag | Description |
| --- | --- |
| `--version` | Show the current ULDAS version and exit |
| `--config PATH` | Path to the configuration file (default: `config/config.yml`) |
| `--create-config` | Create a sample configuration file and exit |
| `--find-mkv` | Locate MKVToolNix installation (Windows only) |
| `--clear-tracking` | Clear all file processing tracking data and exit |
| `--skip-update-check` | Skip checking for updates on startup |

<a id="logging--output"></a>
### Logging & Output

| Flag | Description |
| --- | --- |
| `--verbose`, `-v` | Enable verbose/debug output (sets `show_details` to `true`) |
| `--quiet`, `-q` | Suppress most console output (sets `show_details` to `false`) |
| `--show-details` | Show detailed processing information |
| `--no-show-details` | Hide detailed processing information |

<a id="paths--general"></a>
### Paths & General

| Flag | Description |
| --- | --- |
| `--directory DIR [DIR ...]` | Override directory/directories to scan. Accepts multiple paths |
| `--remux-to-mkv` | Remux non-MKV video files to MKV before processing |
| `--no-remux-to-mkv` | Disable remuxing non-MKV files to MKV |
| `--model {tiny,base,small,medium,large}` | Whisper model size to use for audio language detection |
| `--dry-run` | Simulate all changes without modifying any files |
| `--temp-dir DIR` | Override the temporary directory used for intermediate files |

<a id="voice-activity-detection-vad"></a>
### Voice Activity Detection (VAD)

| Flag | Description |
| --- | --- |
| `--vad` | Enable VAD filter (default) |
| `--no-vad` | Disable VAD filter |
| `--vad-min-speech-duration-ms MS` | Minimum speech duration in milliseconds for VAD |
| `--vad-max-speech-duration-s S` | Maximum speech duration in seconds for VAD |

<a id="device--compute"></a>
### Device & Compute

| Flag | Description |
| --- | --- |
| `--device {auto,cpu,cuda}` | Device for Whisper inference |
| `--compute-type {auto,int8,int8_float16,int16,float16,float32}` | Compute type for Whisper inference |
| `--cpu-threads N` | Number of CPU threads (`0` = auto) |

<a id="confidence--reprocessing"></a>
### Confidence & Reprocessing

| Flag | Description |
| --- | --- |
| `--confidence-threshold F` | Confidence threshold for audio language detection (`0.0`��`1.0`) |
| `--reprocess-all` | Reprocess all audio tracks, even those already tagged |
| `--force-reprocess` | Force reprocessing, ignoring the tracking cache entirely |

<a id="tracking"></a>
### Tracking

| Flag | Description |
| --- | --- |
| `--tracking` | Enable file processing tracking (default) |
| `--no-tracking` | Disable file processing tracking |

<a id="subtitle-processing"></a>
### Subtitle Processing

| Flag | Description |
| --- | --- |
| `--process-subtitles` | Process embedded subtitle tracks |
| `--no-process-subtitles` | Disable processing of embedded subtitle tracks |
| `--process-external-subtitles` | Process external subtitle files |
| `--no-process-external-subtitles` | Disable processing of external subtitle files |
| `--analyze-forced` | Analyze and tag forced subtitles |
| `--no-analyze-forced` | Disable forced subtitle analysis |
| `--detect-sdh` | Detect and tag SDH (Subtitles for the Deaf and Hard of Hearing) |
| `--no-sdh-detection` | Disable SDH subtitle detection |
| `--subtitle-confidence-threshold F` | Confidence threshold for subtitle language detection (`0.0`–`1.0`) |
| `--reprocess-all-subtitles` | Reprocess all subtitle tracks, even those already tagged |

<a id="timeouts"></a>
### Timeouts

| Flag | Description |
| --- | --- |
| `--operation-timeout S` | Timeout in seconds for long operations such as full audio track extraction |

<a id="forced-subtitle-thresholds"></a>
### Forced Subtitle Thresholds

| Flag | Description |
| --- | --- |
| `--forced-sub-low-coverage F` | Low coverage threshold (%) — below this, subtitles are likely forced |
| `--forced-sub-high-coverage F` | High coverage threshold (%) — above this, subtitles are likely full |
| `--forced-sub-low-density F` | Low density threshold (subtitles/min) — below this, likely forced |
| `--forced-sub-high-density F` | High density threshold (subtitles/min) — above this, likely full |
| `--forced-sub-min-count N` | Minimum subtitle count — below this, likely forced |
| `--forced-sub-max-count N` | Maximum subtitle count — above this, likely full |

<a id="cli--config-file-mapping"></a>
### CLI ↔ Config File Mapping

Every CLI option corresponds to a key in the YAML configuration file. CLI arguments always take precedence over config file values. For boolean options with toggle pairs (`--flag` / `--no-flag`), the positive flag wins if both are specified.

| CLI Flag | Config Key | Default |
| --- | --- | --- |
| `--directory` | `path` | `["."]` |
| `--remux-to-mkv` | `remux_to_mkv` | `false` |
| `--show-details` | `show_details` | `true` |
| `--model` | `whisper_model` | `base` |
| `--dry-run` | `dry_run` | `false` |
| `--temp-dir` | `temp_dir` | `""` |
| `--vad` / `--no-vad` | `vad_filter` | `true` |
| `--vad-min-speech-duration-ms` | `vad_min_speech_duration_ms` | `250` |
| `--vad-max-speech-duration-s` | `vad_max_speech_duration_s` | `30` |
| `--device` | `device` | `auto` |
| `--compute-type` | `compute_type` | `auto` |
| `--cpu-threads` | `cpu_threads` | `0` |
| `--confidence-threshold` | `confidence_threshold` | `0.9` |
| `--reprocess-all` | `reprocess_all` | `false` |
| `--force-reprocess` | `force_reprocess` | `false` |
| `--tracking` / `--no-tracking` | `use_tracking` | `true` |
| `--process-subtitles` | `process_subtitles` | `false` |
| `--process-external-subtitles` | `process_external_subtitles` | `false` |
| `--analyze-forced` | `analyze_forced_subtitles` | `false` |
| `--detect-sdh` / `--no-sdh-detection` | `detect_sdh_subtitles` | `true` |
| `--subtitle-confidence-threshold` | `subtitle_confidence_threshold` | `0.85` |
| `--reprocess-all-subtitles` | `reprocess_all_subtitles` | `false` |
| `--operation-timeout` | `operation_timeout_seconds` | `600` |
| `--forced-sub-low-coverage` | `forced_subtitle_low_coverage_threshold` | `25.0` |
| `--forced-sub-high-coverage` | `forced_subtitle_high_coverage_threshold` | `50.0` |
| `--forced-sub-low-density` | `forced_subtitle_low_density_threshold` | `3.0` |
| `--forced-sub-high-density` | `forced_subtitle_high_density_threshold` | `8.0` |
| `--forced-sub-min-count` | `forced_subtitle_min_count_threshold` | `50` |
| `--forced-sub-max-count` | `forced_subtitle_max_count_threshold` | `300` |

***

<a id="example-run-summary"></a>
## 🏞️ Example run summary:

<a id="audio-processing"></a>
### Audio Processing
<img width="926" height="428" alt="Image" src="https://github.com/user-attachments/assets/202d77c9-02ed-4541-ab9b-84d234248961" /><br>
- example run with reprocess_all: true: Samsara is indeed a documentary without spoken dialogue.

<a id="subtitle-processing-1"></a>
### Subtitle Processing
<img width="756" height="142" alt="Image" src="https://github.com/user-attachments/assets/4b1d69ab-f7e7-462e-a7b7-e8f7ec9e89c4" />

***

<a id="need-help-or-have-feedback"></a>
### ⚠️ Need Help or have Feedback?
- Join our [Discord](https://discord.gg/VBNUJd7tx3)

***

<a id="support-the-project"></a>
### ❤️ Support the Project
If you find this project useful, starring the repository is appreciated! ⭐<br>
Big thanks to [DaLeberkasPepi](https://github.com/DaLeberkasPepi) for extensive testing.

<br/>

[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/neekokeen)