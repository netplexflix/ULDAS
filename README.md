# 🎬 MKV Undefined Audio Language Detector (MUALD) 🗣️

Ever downloaded a movie or TV show only to find the audio tracks are labeled as "undefined" or "unknown" in your media player?</br>
This script solves that problem by:

1. Scanning your video files for audio tracks with undefined audio language
2. Extracting audio samples
3. Using AI speech recognition to detect the language [(OpenAI's Whisper)](https://github.com/openai/whisper)
4. Updating the file metadata with the correct language code

The script optionally remuxes non MKV video formats to MKV first.

![Image](https://github.com/user-attachments/assets/28793dbe-8897-46ae-87f1-4a27f7be4cfb)

Requires 
- [Python >=3.11](https://www.python.org/downloads/)
- [FFmpeg](https://ffmpeg.org/download.html)
- [MKVToolNix](https://mkvtoolnix.download/downloads.html)

---

## 🛠️ Installation

### 1️⃣ Download the script
Clone the repository:
```sh
git clone https://github.com/netplexflix/MKV-Undefined-Audio-Language-Detector.git
cd MKV-Undefined-Audio-Language-Detector
```

![#c5f015](https://placehold.co/15x15/c5f015/c5f015.png) Or simply download by pressing the green 'Code' button above and then 'Download Zip'.

### 2️⃣ Install Dependencies
- Ensure you have [Python](https://www.python.org/downloads/) installed (`>=3.11` recommended)
- Open a Terminal in the script's directory
>[!TIP]
>Windows Users: <br/>
>Go to the script folder (where MUALD.py is).</br>
>Right mouse click on an empty space in the folder and click `Open in Windows Terminal`
- Install the required dependencies:
```sh
pip install -r requirements.txt
```

---

## ⚙️ Configuration
Create a config file with the following command:
```sh
python MUALD.py --create-config
```

and change the values where needed:
```
path: P:\Movies #Main Path for your media
remux_to_mkv: true #change to false if you don't want to process non-MKV files
show_details: false #change to true if you want more details of what's happening
whisper_model: base #see Model Size Guide below
dry_run: false #change to true for a dry run (will show what it would do, without actually altering any files)
```

### Model Size Guide

* **tiny:** Fastest, least accurate
* **base:** Good balance (recommended for most users)
* **small:** More accurate, slower
* **medium:** Very accurate, much slower
* **large:** Most accurate, very slow

---

## 🚀 Usage

Run the script with:
```sh
python MUALD.py
```

> [!TIP]
> Windows users can create a batch file for quick launching:
> ```batch
> "C:\Path\To\Python\python.exe" "Path\To\Script\MUALD.py"
> pause
> ```

---

## 📄 Supported File Formats
Always Processed:
* **MKV files:** Primary target format

With `remux_to_mkv: true`
* MP4, AVI, MOV, WMV, FLV, WebM, M4V, M2TS, MTS, TS, VOB
* Note: Original files are deleted after successful conversion

---

### ⚠️ Need Help or have Feedback?
- Join our [Discord](https://discord.gg/VBNUJd7tx3)

---

### ❤️ Support the Project
If you find this project useful, please ⭐ star the repository and share it!

<br/>

[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/neekokeen)

