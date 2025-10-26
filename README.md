<img width="668" height="201" alt="Image" src="https://github.com/user-attachments/assets/d6d715f7-59cb-4321-ad29-97d04dbd2de5" /> <br>

Do you have Movies or TV shows in your media player for which the audio and/or subtitle tracks are labeled as "undefined" or "unknown"?</br>
ULDAS (Undefined Language Detector for Audio and Subtitles) solves that problem by:

1. Scanning your video files for audio and subtitle tracks with undefined language
2. Extracting audio and subtitle samples
3. Using AI speech recognition to detect the audio language
4. Detecting subtitle language (can also detect if subtitles are [FORCED] and/or [SDH])
5. Updating the file metadata with the correct language codes and flags

The script optionally remuxes non MKV video formats to MKV first.

![Image](https://github.com/user-attachments/assets/28793dbe-8897-46ae-87f1-4a27f7be4cfb)

Requires 
- [Python >=3.11](https://www.python.org/downloads/)
- [FFmpeg](https://ffmpeg.org/download.html)
- [MKVToolNix](https://mkvtoolnix.download/downloads.html)
- [Tesseract-OCR](https://github.com/tesseract-ocr/tesseract?tab=readme-ov-file#installing-tesseract) for image based subtitles (e.g. PGS)

---

## 🛠️ Installation

### 1️⃣ Download the script
Clone the repository:
```sh
git clone https://github.com/netplexflix/ULDAS.git
cd ULDAS
```

![#c5f015](https://placehold.co/15x15/c5f015/c5f015.png) Or simply download by pressing the green 'Code' button above and then 'Download Zip'.

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

---

## ⚙️ Configuration
Rename `config.example.yml` to `config.yml` and change the values where needed:

- **path**: Main Paths for your media.
- **remux_to_mkv**: `true` remuxes non-MKV files so they can be processed too
- **show_details**: `true` will show you more details of what's happening
- **dry_run**: `true` will do a dry run (will show what it would do, without actually altering any files)
- **process_subtitles**: `true` will process undefined subtitle tracks
- **analyze_forced_subtitles**: `true` will analyze whether a subtitle track has "Forced Subtitles" or not
- **detect_sdh_subtitles**: `true` will analyze whether a subtitle track has 'hearing impaired' support. (e.g.: [Dogs barking], [Narrator:],... )

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

### Model Size Guide

* **tiny:** Fastest, least accurate
* **base:** Good balance
* **small:** More accurate, slower (used during development tests)
* **medium:** Very accurate, much slower
* **large:** Most accurate, very slow

---

## 🚀 Usage

Run the script with:
```sh
python ULDAS.py
```

> [!TIP]
> Windows users can create a batch file for quick launching:
> ```batch
> "C:\Path\To\Python\python.exe" "Path\To\Script\ULDAS.py"
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

---

## 📄 Supported File Formats
Always Processed:
* **MKV files:** Primary target format

With `remux_to_mkv: true`
* MP4, AVI, MOV, WMV, FLV, WebM, M4V, M2TS, MTS, TS, VOB
* Note: Original files are deleted after successful conversion

---

## 🏞️ Example run summary:
### Audio Processing
<img width="926" height="428" alt="Image" src="https://github.com/user-attachments/assets/202d77c9-02ed-4541-ab9b-84d234248961" /><br>
* example run with reprocess_all: true: Samsara is indeed a documentary without spoken dialogue.

### Subtitle Processing
<img width="756" height="142" alt="Image" src="https://github.com/user-attachments/assets/4b1d69ab-f7e7-462e-a7b7-e8f7ec9e89c4" />

---

### ⚠️ Need Help or have Feedback?
- Join our [Discord](https://discord.gg/VBNUJd7tx3)

---

### ❤️ Support the Project
If you find this project useful, starring the repository is appreciated! ⭐<br>
Big thanks to [DaLeberkasPepi](https://github.com/DaLeberkasPepi) for extensive testing.

<br/>

[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/neekokeen)