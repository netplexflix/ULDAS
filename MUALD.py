import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated.*", category=UserWarning, module="ctranslate2.*")
import os
import sys
import subprocess
import tempfile
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
from faster_whisper import WhisperModel
import logging
import shutil
import yaml
import time
import requests
from packaging import version
import psutil

VERSION = '2.1'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_cpu_limits():
    try:
        current_process = psutil.Process()
        
        if hasattr(os, 'nice'):
            os.nice(10)  # Unix/Linux: increase nice value (lower priority)
        elif hasattr(current_process, 'nice'):
            current_process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)  # Windows
        
        # Limit to 75% of available CPU cores
        cpu_count = psutil.cpu_count()
        max_cores = max(1, int(cpu_count * 0.75))
        
        if hasattr(current_process, 'cpu_affinity'):
            available_cores = list(range(min(max_cores, cpu_count)))
            current_process.cpu_affinity(available_cores)
            logger.info(f"Limited to {len(available_cores)} of {cpu_count} CPU cores")
        
    except Exception as e:
        logger.warning(f"Could not set CPU limits: {e}")

def limit_subprocess_resources(cmd):
    if sys.platform == 'win32':
        return cmd
    else:
        return ['nice', '-n', '10'] + cmd

class Config:
    def __init__(self):
        self.path = ["."]
        self.remux_to_mkv = False
        self.show_details = True
        self.whisper_model = "base"
        self.dry_run = False
        self.vad_filter = True
        self.vad_min_speech_duration_ms = 250
        self.vad_max_speech_duration_s = 30
        self.device = "auto"  # auto, cpu, cuda, or specific device
        self.compute_type = "auto"  # auto, int8, int8_float16, int16, float16, float32
        self.cpu_threads = 0  # 0 for auto
        self.confidence_threshold = 0.9
        self.reprocess_all = False  # reprocess all audio tracks instead of only undefined ones
        
    def load_from_file(self, config_path: str = "config/config.yml"):
        config_dir = os.path.dirname(config_path)
        if config_dir and not os.path.exists(config_dir):
            os.makedirs(config_dir, exist_ok=True)
            
        if not os.path.exists(config_path):
            logger.info(f"Config file {config_path} not found, using defaults")
            return
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
                
            if config_data:
                self.path = config_data.get('path', self.path)
                self.remux_to_mkv = config_data.get('remux_to_mkv', self.remux_to_mkv)
                self.show_details = config_data.get('show_details', self.show_details)
                self.whisper_model = config_data.get('whisper_model', self.whisper_model)
                self.dry_run = config_data.get('dry_run', self.dry_run)
                self.vad_filter = config_data.get('vad_filter', self.vad_filter)
                self.vad_min_speech_duration_ms = config_data.get('vad_min_speech_duration_ms', self.vad_min_speech_duration_ms)
                self.vad_max_speech_duration_s = config_data.get('vad_max_speech_duration_s', self.vad_max_speech_duration_s)
                self.device = config_data.get('device', self.device)
                self.compute_type = config_data.get('compute_type', self.compute_type)
                self.cpu_threads = config_data.get('cpu_threads', self.cpu_threads)
                self.confidence_threshold = config_data.get('confidence_threshold', self.confidence_threshold)
                self.reprocess_all = config_data.get('reprocess_all', self.reprocess_all)
                
            logger.info(f"Configuration loaded from {config_path}")
            
        except Exception as e:
            logger.error(f"Error loading config file {config_path}: {e}")
            logger.info("Using default configuration")
    
    def create_sample_config(self, config_path: str = "config/config.yml"):
        sample_config = {
            'path': [
                "P:/Movies",
                "P:/TV"
            ],
            'remux_to_mkv': True,
            'show_details': False,
            'whisper_model': 'small',
            'dry_run': False,
            'vad_filter': True,
            'vad_min_speech_duration_ms': 250,
            'vad_max_speech_duration_s': 30,
            'device': 'auto',
            'compute_type': 'auto',
            'cpu_threads': 0,
            'confidence_threshold': 0.9,
            'reprocess_all': False
        }
        
        try:
            config_dir = os.path.dirname(config_path)
            if config_dir and not os.path.exists(config_dir):
                os.makedirs(config_dir, exist_ok=True)
                
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(sample_config, f, default_flow_style=False, sort_keys=False)
            print(f"Sample configuration file created: {config_path}")
            print("Edit this file to customize your settings")
        except Exception as e:
            logger.error(f"Error creating config file: {e}")

def find_executable(name: str) -> Optional[str]:
    if shutil.which(name):
        return name
    
    if sys.platform == 'win32':
        exe_name = f"{name}.exe"
        if shutil.which(exe_name):
            return exe_name
        
        if name == 'mkvpropedit':
            possible_paths = [
                "C:\\Program Files\\MKVToolNix\\mkvpropedit.exe",
                "C:\\Program Files (x86)\\MKVToolNix\\mkvpropedit.exe",
                "C:\\ProgramData\\chocolatey\\lib\\mkvtoolnix\\tools\\mkvpropedit.exe",
                "C:\\MKVToolNix\\mkvpropedit.exe",
                "C:\\Tools\\MKVToolNix\\mkvpropedit.exe",
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    if logger.isEnabledFor(logging.INFO):
                        logger.info(f"Found mkvpropedit at: {path}")
                    return path
            
            # Try to find MKVToolNix folder in Program Files
            program_files = ["C:\\Program Files", "C:\\Program Files (x86)"]
            for pf in program_files:
                if os.path.exists(pf):
                    for item in os.listdir(pf):
                        if "mkv" in item.lower():
                            potential_path = os.path.join(pf, item, "mkvpropedit.exe")
                            if os.path.exists(potential_path):
                                if logger.isEnabledFor(logging.INFO):
                                    logger.info(f"Found mkvpropedit at: {potential_path}")
                                return potential_path
        
        # Also check common installation paths for other tools
        common_paths = [
            f"C:\\Program Files\\FFmpeg\\bin\\{exe_name}",
            f"C:\\Program Files (x86)\\FFmpeg\\bin\\{exe_name}",
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                return path
    
    return None

def find_mkvtoolnix_installation():
    if sys.platform != 'win32':
        return
    
    print("Searching for MKVToolNix installation...")
    
    try:
        import winreg
        
        reg_paths = [
            r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall",
            r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"
        ]
        
        for reg_path in reg_paths:
            try:
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, reg_path) as key:
                    i = 0
                    while True:
                        try:
                            subkey_name = winreg.EnumKey(key, i)
                            with winreg.OpenKey(key, subkey_name) as subkey:
                                try:
                                    display_name = winreg.QueryValueEx(subkey, "DisplayName")[0]
                                    if "mkvtoolnix" in display_name.lower():
                                        try:
                                            install_location = winreg.QueryValueEx(subkey, "InstallLocation")[0]
                                            print(f"Found MKVToolNix installed at: {install_location}")
                                            mkvpropedit_path = os.path.join(install_location, "mkvpropedit.exe")
                                            if os.path.exists(mkvpropedit_path):
                                                print(f"mkvpropedit.exe found at: {mkvpropedit_path}")
                                                return mkvpropedit_path
                                        except FileNotFoundError:
                                            pass
                                except FileNotFoundError:
                                    pass
                            i += 1
                        except OSError:
                            break
            except OSError:
                continue
                
    except ImportError:
        pass
    
    search_dirs = [
        "C:\\Program Files",
        "C:\\Program Files (x86)",
        "C:\\ProgramData\\chocolatey\\lib"
    ]
    
    for base_dir in search_dirs:
        if os.path.exists(base_dir):
            for item in os.listdir(base_dir):
                if "mkv" in item.lower():
                    full_path = os.path.join(base_dir, item)
                    print(f"Found MKV-related directory: {full_path}")
                    
                    possible_exe_paths = [
                        os.path.join(full_path, "mkvpropedit.exe"),
                        os.path.join(full_path, "tools", "mkvpropedit.exe"),
                        os.path.join(full_path, "bin", "mkvpropedit.exe")
                    ]
                    
                    for exe_path in possible_exe_paths:
                        if os.path.exists(exe_path):
                            print(f"mkvpropedit.exe found at: {exe_path}")
                            return exe_path
    
    return None

class MKVLanguageDetector:
    def __init__(self, config: Config):
        setup_cpu_limits()
        
        self.config = config
        
        # Determine device and compute type
        device = self._determine_device()
        compute_type = self._determine_compute_type(device)
        cpu_threads = config.cpu_threads if config.cpu_threads > 0 else 0
        
        if config.show_details:
            logger.info(f"Initializing faster-whisper with device: {device}, compute_type: {compute_type}")
        
        # Initialize faster-whisper model
        try:
            self.whisper_model = WhisperModel(
                config.whisper_model, 
                device=device, 
                compute_type=compute_type,
                cpu_threads=cpu_threads,
                download_root=None,
                local_files_only=False
            )
        except Exception as e:
            logger.warning(f"Failed to initialize with preferred settings: {e}")
            # Fallback to CPU with default settings
            self.whisper_model = WhisperModel(config.whisper_model, device="cpu")
            logger.info("Fallback: Using CPU with default settings")
        
        self.ffmpeg = find_executable('ffmpeg')
        self.ffprobe = find_executable('ffprobe') 
        self.mkvpropedit = find_executable('mkvpropedit')
        self.mkvmerge = find_executable('mkvmerge')
        
        if not all([self.ffmpeg, self.ffprobe, self.mkvpropedit]):
            missing = []
            if not self.ffmpeg: missing.append('ffmpeg')
            if not self.ffprobe: missing.append('ffprobe')
            if not self.mkvpropedit: missing.append('mkvpropedit')
            raise RuntimeError(f"Missing executables: {', '.join(missing)}")
        
        if not self.mkvmerge:
            logger.warning("mkvmerge not found - language detection may be less accurate")
            logger.warning("Install MKVToolNix for better compatibility: https://mkvtoolnix.download/")
        
        self.language_codes = {
            'english': 'eng',
            'spanish': 'spa',
            'french': 'fre',
            'german': 'ger',
            'italian': 'ita',
            'portuguese': 'por',
            'russian': 'rus',
            'japanese': 'jpn',
            'chinese': 'chi',
            'korean': 'kor',
            'arabic': 'ara',
            'hindi': 'hin',
            'dutch': 'dut',
            'swedish': 'swe',
            'norwegian': 'nor',
            'danish': 'dan',
            'finnish': 'fin',
            'polish': 'pol',
            'czech': 'cze',
            'hungarian': 'hun',
            'greek': 'gre',
            'turkish': 'tur',
            'hebrew': 'heb',
            'thai': 'tha',
            'vietnamese': 'vie',
            'ukrainian': 'ukr',
            'bulgarian': 'bul',
            'romanian': 'rum',
            'slovak': 'slo',
            'slovenian': 'slv',
            'serbian': 'srp',
            'croatian': 'hrv',
            'bosnian': 'bos',
            'albanian': 'alb',
            'macedonian': 'mac',
            'lithuanian': 'lit',
            'latvian': 'lav',
            'estonian': 'est',
            'maltese': 'mlt',
            'icelandic': 'ice',
            'irish': 'gle',
            'welsh': 'wel',
            'basque': 'baq',
            'catalan': 'cat',
            'galician': 'glg',
            'persian': 'per',
            'urdu': 'urd',
            'bengali': 'ben',
            'gujarati': 'guj',
            'punjabi': 'pan',
            'tamil': 'tam',
            'telugu': 'tel',
            'kannada': 'kan',
            'malayalam': 'mal',
            'marathi': 'mar',
            'nepali': 'nep',
            'sinhalese': 'sin',
            'burmese': 'bur',
            'khmer': 'khm',
            'lao': 'lao',
            'tibetan': 'tib',
            'mongolian': 'mon',
            'kazakh': 'kaz',
            'uzbek': 'uzb',
            'kyrgyz': 'kir',
            'tajik': 'tgk',
            'turkmen': 'tuk',
            'azerbaijani': 'aze',
            'armenian': 'arm',
            'georgian': 'geo',
            'amharic': 'amh',
            'swahili': 'swa',
            'yoruba': 'yor',
            'igbo': 'ibo',
            'hausa': 'hau',
            'somali': 'som',
            'afrikaans': 'afr',
            'zulu': 'zul',
            'xhosa': 'xho',
            'malay': 'may',
            'indonesian': 'ind',
            'tagalog': 'tgl',
            'cebuano': 'ceb',
            'javanese': 'jav',
            'sundanese': 'sun',
            'esperanto': 'epo',
            'latin': 'lat',
            'mandarin': 'chi',
            'cantonese': 'chi',
            'simplified chinese': 'chi',
            'traditional chinese': 'chi',
            'farsi': 'per',
            'filipino': 'tgl',
            'bahasa indonesia': 'ind',
            'bahasa malaysia': 'may',
            'no linguistic content': 'zxx'
        }
        
        self.video_extensions = {'.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.m2ts', '.mts', '.ts', '.vob'}

    def normalize_language_code(self, lang_code: str) -> str:
        if not lang_code:
            return 'und'
        
        lang_code = lang_code.lower().strip()
        
        if lang_code in ['', 'und', 'unknown', 'undefined', 'undetermined']:
            return 'und'
        
        if lang_code == 'zxx':
            return 'zxx'
        
        # ISO 639-2 to ISO 639-1 mapping (normalize to 2-letter codes)
        iso639_2_to_1_mapping = {
            'ger': 'de',  # German
            'eng': 'en',  # English
            'spa': 'es',  # Spanish
            'fre': 'fr',  # French
            'ita': 'it',  # Italian
            'por': 'pt',  # Portuguese
            'rus': 'ru',  # Russian
            'jpn': 'ja',  # Japanese
            'kor': 'ko',  # Korean
            'chi': 'zh',  # Chinese
            'ara': 'ar',  # Arabic
            'hin': 'hi',  # Hindi
            'dut': 'nl',  # Dutch
            'swe': 'sv',  # Swedish
            'nor': 'no',  # Norwegian
            'dan': 'da',  # Danish
            'fin': 'fi',  # Finnish
            'pol': 'pl',  # Polish
            'cze': 'cs',  # Czech
            'hun': 'hu',  # Hungarian
            'gre': 'el',  # Greek
            'tur': 'tr',  # Turkish
            'heb': 'he',  # Hebrew
            'tha': 'th',  # Thai
            'vie': 'vi',  # Vietnamese
            'ukr': 'uk',  # Ukrainian
            'bul': 'bg',  # Bulgarian
            'rum': 'ro',  # Romanian
            'slo': 'sk',  # Slovak
            'slv': 'sl',  # Slovenian
            'srp': 'sr',  # Serbian
            'hrv': 'hr',  # Croatian
            'bos': 'bs',  # Bosnian
            'alb': 'sq',  # Albanian
            'mac': 'mk',  # Macedonian
            'lit': 'lt',  # Lithuanian
            'lav': 'lv',  # Latvian
            'est': 'et',  # Estonian
            'mlt': 'mt',  # Maltese
            'ice': 'is',  # Icelandic
            'gle': 'ga',  # Irish
            'wel': 'cy',  # Welsh
            'baq': 'eu',  # Basque
            'cat': 'ca',  # Catalan
            'glg': 'gl',  # Galician
            'per': 'fa',  # Persian
            'urd': 'ur',  # Urdu
            'ben': 'bn',  # Bengali
            'guj': 'gu',  # Gujarati
            'pan': 'pa',  # Punjabi
            'tam': 'ta',  # Tamil
            'tel': 'te',  # Telugu
            'kan': 'kn',  # Kannada
            'mal': 'ml',  # Malayalam
            'mar': 'mr',  # Marathi
            'nep': 'ne',  # Nepali
            'sin': 'si',  # Sinhalese
            'bur': 'my',  # Burmese
            'khm': 'km',  # Khmer
            'lao': 'lo',  # Lao
            'tib': 'bo',  # Tibetan
            'mon': 'mn',  # Mongolian
            'kaz': 'kk',  # Kazakh
            'uzb': 'uz',  # Uzbek
            'kir': 'ky',  # Kyrgyz
            'tgk': 'tg',  # Tajik
            'tuk': 'tk',  # Turkmen
            'aze': 'az',  # Azerbaijani
            'arm': 'hy',  # Armenian
            'geo': 'ka',  # Georgian
            'amh': 'am',  # Amharic
            'swa': 'sw',  # Swahili
            'yor': 'yo',  # Yoruba
            'ibo': 'ig',  # Igbo
            'hau': 'ha',  # Hausa
            'som': 'so',  # Somali
            'afr': 'af',  # Afrikaans
            'zul': 'zu',  # Zulu
            'xho': 'xh',  # Xhosa
            'may': 'ms',  # Malay
            'ind': 'id',  # Indonesian
            'tgl': 'tl',  # Tagalog
            'jav': 'jv',  # Javanese
            'sun': 'su',  # Sundanese
            'epo': 'eo',  # Esperanto
            'lat': 'la',  # Latin
        }
        
        # If it's a 3-letter code, convert to 2-letter
        if lang_code in iso639_2_to_1_mapping:
            return iso639_2_to_1_mapping[lang_code]
        
        # Handle alternative 3-letter codes
        alternative_codes = {
            'deu': 'de',  # Alternative German code
            'fra': 'fr',  # Alternative French code
            'nld': 'nl',  # Alternative Dutch code
            'ces': 'cs',  # Alternative Czech code
            'slk': 'sk',  # Alternative Slovak code
            'ron': 'ro',  # Alternative Romanian code
        }
        
        if lang_code in alternative_codes:
            return alternative_codes[lang_code]
        
        # If it's already a 2-letter code, return as-is
        if len(lang_code) == 2:
            return lang_code
        
        # If we can't normalize it, return as-is
        return lang_code
		
    def _determine_device(self):
        if self.config.device != "auto":
            return self.config.device
        
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        
        return "cpu"
    
    def _determine_compute_type(self, device):
        if self.config.compute_type != "auto":
            return self.config.compute_type
        
        if device == "cuda":
            return "float16"
        else:
            return "int8"
    
    def find_video_files(self, directory: str) -> List[Path]:
        directory_path = Path(directory)
        video_files = []
        
        extensions = {'.mkv'}
        
        if self.config.remux_to_mkv:
            extensions.update(self.video_extensions)
        
        for ext in extensions:
            for file_path in directory_path.rglob(f"*{ext}"):
                if file_path.is_file():
                    video_files.append(file_path)
        
        if self.config.show_details:
            logger.info(f"Found {len(video_files)} video files")
        
        return video_files
    
    def remux_to_mkv(self, file_path: Path) -> Optional[Path]:
        if file_path.suffix.lower() == '.mkv':
            return file_path
        
        mkv_path = file_path.with_suffix('.mkv')
        
        if mkv_path.exists():
            if self.config.show_details:
                logger.info(f"MKV version already exists: {mkv_path.name}")
            return mkv_path
        
        if self.config.dry_run:
            print(f"[DRY RUN] Would remux {file_path.name} to {mkv_path.name} and remove original")
            return mkv_path
        
        try:
            print(f"Remuxing {file_path.name} to MKV format...")
            
            analyze_cmd = [
                self.ffprobe, '-v', 'quiet', '-print_format', 'json',
                '-show_streams', str(file_path)
            ]
            
            try:
                analyze_result = subprocess.run(analyze_cmd, check=True, capture_output=True, text=True, encoding='utf-8', errors='replace')
                stream_info = json.loads(analyze_result.stdout)
                streams = stream_info.get('streams', [])
            except (subprocess.CalledProcessError, json.JSONDecodeError, UnicodeDecodeError) as e:
                logger.warning(f"Could not analyze input streams: {e}")
                streams = []
            
            # Check for problematic M2TS characteristics
            is_m2ts = file_path.suffix.lower() in ['.m2ts', '.mts', '.ts']
            has_pcm_bluray = any(stream.get('codec_name') == 'pcm_bluray' for stream in streams if stream.get('codec_type') == 'audio')
            
            map_args = []
            has_video = False
            has_audio = False
            
            for i, stream in enumerate(streams):
                codec_type = stream.get('codec_type', '')
                codec_name = stream.get('codec_name', '')
                
                if codec_type == 'video':
                    map_args.extend(['-map', f'0:{i}'])
                    has_video = True
                elif codec_type == 'audio':
                    map_args.extend(['-map', f'0:{i}'])
                    has_audio = True
                elif codec_type == 'subtitle':
                    supported_subtitle_codecs = [
                        'subrip', 'srt', 'ass', 'ssa', 'webvtt', 'mov_text',
                        'pgs', 'dvdsub', 'dvbsub', 'hdmv_pgs_subtitle'
                    ]
                    
                    if codec_name.lower() in supported_subtitle_codecs:
                        map_args.extend(['-map', f'0:{i}'])
                        if self.config.show_details:
                            logger.info(f"Including subtitle stream {i} ({codec_name})")
                    else:
                        if self.config.show_details:
                            logger.info(f"Skipping unsupported subtitle stream {i} ({codec_name})")
            
            if not map_args:
                map_args = ['-map', '0:v', '-map', '0:a']
                if self.config.show_details:
                    logger.info("Using fallback mapping (video and audio only)")
            
            # Define remux strategies with M2TS-specific handling
            remux_strategies = []
            
            # Strategy 1: M2TS-specific strategy with timestamp fixes and audio conversion
            if is_m2ts:
                remux_strategies.append({
                    'name': 'm2ts_optimized',
                    'args': [
                        self.ffmpeg, '-y', '-v', 'warning', '-fflags', '+genpts',
                        '-analyzeduration', '100M', '-probesize', '100M',
                        '-i', str(file_path),
                        '-map', '0:v', '-c:v', 'copy',
                        '-map', '0:a', '-c:a', 'flac' if has_pcm_bluray else 'copy',
                        '-avoid_negative_ts', 'make_zero',
                        '-fflags', '+discardcorrupt',
                        '-map_metadata', '0',
                        str(mkv_path)
                    ]
                })
                
                # Strategy 2: M2TS with PCM to AC3 conversion for better compatibility
                if has_pcm_bluray:
                    remux_strategies.append({
                        'name': 'm2ts_pcm_to_ac3',
                        'args': [
                            self.ffmpeg, '-y', '-v', 'warning', '-fflags', '+genpts',
                            '-analyzeduration', '100M', '-probesize', '100M',
                            '-i', str(file_path),
                            '-map', '0:v', '-c:v', 'copy',
                            '-map', '0:a', '-c:a', 'ac3', '-b:a', '640k',
                            '-avoid_negative_ts', 'make_zero',
                            '-fflags', '+discardcorrupt',
                            '-map_metadata', '0',
                            str(mkv_path)
                        ]
                    })
            
            # Strategy 3: Enhanced selective copy with better error handling
            remux_strategies.append({
                'name': 'selective_copy_enhanced',
                'args': [
                    self.ffmpeg, '-y', '-v', 'warning', '-fflags', '+genpts',
                    '-analyzeduration', '50M', '-probesize', '50M',
                    '-i', str(file_path),
                    '-c', 'copy'
                ] + map_args + [
                    '-avoid_negative_ts', 'make_zero',
                    '-fflags', '+discardcorrupt',
                    '-map_metadata', '0',
                    str(mkv_path)
                ]
            })
            
            # Strategy 4: Original selective copy
            remux_strategies.append({
                'name': 'selective_copy',
                'args': [
                    self.ffmpeg, '-y', '-v', 'warning', '-fflags', '+genpts',
                    '-i', str(file_path),
                    '-c', 'copy'
                ] + map_args + [
                    '-avoid_negative_ts', 'make_zero',
                    '-map_metadata', '0',
                    str(mkv_path)
                ]
            })
            
            # Strategy 5: Convert subtitles
            remux_strategies.append({
                'name': 'convert_subtitles',
                'args': [
                    self.ffmpeg, '-y', '-v', 'warning', '-fflags', '+genpts',
                    '-i', str(file_path),
                    '-map', '0:v', '-c:v', 'copy',
                    '-map', '0:a', '-c:a', 'copy',
                    '-map', '0:s?', '-c:s', 'srt',  # Convert subtitles to SRT
                    '-avoid_negative_ts', 'make_zero',
                    '-map_metadata', '0',
                    str(mkv_path)
                ]
            })
            
            # Strategy 6: Video and audio only (no subtitles)
            remux_strategies.append({
                'name': 'no_subtitles',
                'args': [
                    self.ffmpeg, '-y', '-v', 'warning', '-fflags', '+genpts',
                    '-i', str(file_path),
                    '-map', '0:v', '-c:v', 'copy',
                    '-map', '0:a', '-c:a', 'copy',
                    '-avoid_negative_ts', 'make_zero',
                    '-map_metadata', '0',
                    str(mkv_path)
                ]
            })
            
            # Strategy 7: Force container without stream copy (slower but more compatible)
            remux_strategies.append({
                'name': 'force_remux',
                'args': [
                    self.ffmpeg, '-y', '-v', 'warning',
                    '-i', str(file_path),
                    '-map', '0:v', '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23',
                    '-map', '0:a', '-c:a', 'copy',
                    '-avoid_negative_ts', 'make_zero',
                    str(mkv_path)
                ]
            })
            
            last_error = None
            remux_successful = False
            
            for strategy in remux_strategies:
                try:
                    if self.config.show_details:
                        logger.info(f"Trying remux strategy: {strategy['name']}")
                    
                    limited_cmd = limit_subprocess_resources(strategy['args'])
                    
                    result = subprocess.run(limited_cmd, check=True, capture_output=True, 
                                          text=True, encoding='utf-8', errors='replace')
                    
                    # Verify the output file
                    if mkv_path.exists() and mkv_path.stat().st_size > 10000:
                        # Verify the remuxed file is valid
                        try:
                            verify_cmd = [
                                self.ffprobe, '-v', 'quiet', '-print_format', 'json',
                                '-show_streams', str(mkv_path)
                            ]
                            verify_result = subprocess.run(verify_cmd, check=True, capture_output=True, text=True, encoding='utf-8', errors='replace')
                            json.loads(verify_result.stdout)  # Verify JSON is valid
                            
                            # Mark as successful and break out of strategy loop
                            remux_successful = True
                            if self.config.show_details:
                                logger.info(f"Successfully remuxed with strategy '{strategy['name']}'")
                            else:
                                print(f"Successfully remuxed to: {mkv_path.name}")
                            break
                            
                        except (subprocess.CalledProcessError, json.JSONDecodeError, UnicodeDecodeError) as e:
                            logger.warning(f"Strategy '{strategy['name']}' produced invalid file: {e}")
                            if mkv_path.exists():
                                mkv_path.unlink()
                            continue
                    else:
                        logger.warning(f"Strategy '{strategy['name']}' produced empty or very small file")
                        if mkv_path.exists():
                            mkv_path.unlink()
                        continue
                        
                except subprocess.CalledProcessError as e:
                    last_error = getattr(e, 'stderr', str(e))
                    if self.config.show_details:
                        logger.debug(f"Strategy '{strategy['name']}' failed: {last_error}")
                    # Clean up any partial file
                    if mkv_path.exists():
                        mkv_path.unlink()
                    continue
            
            # If remux was successful, remove the original file
            if remux_successful:
                try:
                    import gc
                    gc.collect()
                    
                    # Add 10 second delay to ensure Windows releases file handles
                    time.sleep(10)
                    
                    # Try multiple times with increasing delays
                    for attempt in range(3):
                        try:
                            file_path.unlink()
                            if self.config.show_details:
                                logger.info(f"Removed original file: {file_path.name}")
                            return mkv_path
                        except (OSError, PermissionError) as e:
                            if attempt < 2:
                                if self.config.show_details:
                                    logger.debug(f"File deletion attempt {attempt + 1} failed, retrying: {e}")
                                time.sleep(1.0 * (attempt + 1))
                            else:
                                raise e
                                
                except Exception as e:
                    # Store the deletion failure info for later reporting
                    if not hasattr(self, 'deletion_failures'):
                        self.deletion_failures = []
                    self.deletion_failures.append({
                        'original_file': str(file_path),
                        'mkv_file': str(mkv_path),
                        'error': str(e)
                    })
                    
                    logger.warning(f"Remux successful but failed to remove original file {file_path}: {e}")
                    if self.config.show_details:
                        logger.warning("You may need to manually delete the original file")
                    return mkv_path
            else:
                # All strategies failed
                logger.error(f"All remux strategies failed for {file_path}")
                if last_error:
                    logger.error(f"Last error: {last_error}")
                return None
            
        except Exception as e:
            logger.error(f"Unexpected error during remux: {e}")
            # Clean up any partial file
            if mkv_path.exists():
                mkv_path.unlink()
            return None
    
    def get_mkv_info(self, file_path: Path) -> Dict:
        """Get MKV info using mkvmerge -J"""
        try:
            # First try to find mkvmerge
            mkvmerge = find_executable('mkvmerge')
            if not mkvmerge:
                # Fall back to ffprobe if mkvmerge is not available
                logger.warning("mkvmerge not found, falling back to ffprobe (may not show accurate language info)")
                return self._get_mkv_info_ffprobe(file_path)
            
            cmd = [mkvmerge, '-J', str(file_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding='utf-8', errors='replace')
            mkvmerge_data = json.loads(result.stdout)
            
            # Convert mkvmerge format to ffprobe-like format for compatibility
            converted_info = {'streams': []}
            
            for track in mkvmerge_data.get('tracks', []):
                props = track.get('properties', {})
                stream_info = {
                    'index': track.get('id', 0),
                    'codec_type': self._convert_track_type(track.get('type', '')),
                    'codec_name': props.get('codec_id', ''),
                    'tags': {}
                }
                
                # Get language from track properties (what media players actually use)
                language = props.get('language', '')
                if language:
                    stream_info['tags']['language'] = language
                    
                # Also include track name if present
                track_name = props.get('track_name', '')
                if track_name:
                    stream_info['tags']['title'] = track_name
                    
                converted_info['streams'].append(stream_info)
                
            return converted_info
            
        except subprocess.CalledProcessError as e:
            try:
                # Try with different encoding
                result = subprocess.run(cmd, capture_output=True, check=True, encoding='cp1252', errors='replace')
                mkvmerge_data = json.loads(result.stdout.decode('utf-8', errors='replace') if isinstance(result.stdout, bytes) else result.stdout)
                # Process the same way as above...
                converted_info = {'streams': []}
                for track in mkvmerge_data.get('tracks', []):
                    props = track.get('properties', {})
                    stream_info = {
                        'index': track.get('id', 0),
                        'codec_type': self._convert_track_type(track.get('type', '')),
                        'codec_name': props.get('codec_id', ''),
                        'tags': {}
                    }
                    language = props.get('language', '')
                    if language:
                        stream_info['tags']['language'] = language
                    track_name = props.get('track_name', '')
                    if track_name:
                        stream_info['tags']['title'] = track_name
                    converted_info['streams'].append(stream_info)
                return converted_info
            except (subprocess.CalledProcessError, UnicodeDecodeError, json.JSONDecodeError):
                logger.error(f"Error getting info for {file_path}: Unable to read file metadata with mkvmerge")
                return self._get_mkv_info_ffprobe(file_path)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.error(f"Error parsing mkvmerge output for {file_path}: {e}")
            return self._get_mkv_info_ffprobe(file_path)
    
    def _convert_track_type(self, mkv_type: str) -> str:
        """Convert mkvmerge track type to ffprobe-compatible type"""
        type_mapping = {
            'video': 'video',
            'audio': 'audio', 
            'subtitles': 'subtitle'
        }
        return type_mapping.get(mkv_type.lower(), mkv_type)
    
    def _get_mkv_info_ffprobe(self, file_path: Path) -> Dict:
        """Fallback method using ffprobe"""
        try:
            cmd = [
                self.ffprobe, '-v', 'quiet', '-print_format', 'json',
                '-show_streams', str(file_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding='utf-8', errors='replace')
            return json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            try:
                result = subprocess.run(cmd, capture_output=True, check=True, encoding='cp1252', errors='replace')
                return json.loads(result.stdout.decode('utf-8', errors='replace') if isinstance(result.stdout, bytes) else result.stdout)
            except (subprocess.CalledProcessError, UnicodeDecodeError, json.JSONDecodeError):
                logger.error(f"Error getting info for {file_path}: Unable to read file metadata (encoding issues)")
                return {}
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.error(f"Error parsing ffprobe output for {file_path}: {e}")
            return {}
    
    def find_undefined_audio_tracks(self, file_path: Path) -> List[Tuple[int, Dict]]:
        info = self.get_mkv_info(file_path)
        undefined_tracks = []
        
        if 'streams' not in info:
            return undefined_tracks
        
        audio_track_count = 0
        for i, stream in enumerate(info['streams']):
            if stream.get('codec_type') == 'audio':
                tags = stream.get('tags', {})
                
                # Check multiple possible language tag keys (case-insensitive)
                language = None
                for key in tags:
                    if key.lower() in ['language', 'lang']:
                        language = tags[key].lower().strip()
                        break
                undefined_indicators = ['und', 'unknown', 'undefined', 'undetermined', '']
                
                if not language or language in undefined_indicators:
                    undefined_tracks.append((audio_track_count, stream, i))
                    if self.config.show_details:
                        lang_display = language if language else 'missing'
                        logger.info(f"Found undefined audio track {audio_track_count} (stream {i}) in {file_path.name} - language: '{lang_display}'")
                else:
                    if self.config.show_details:
                        logger.info(f"Audio track {audio_track_count} (stream {i}) already has language: '{language}' - skipping")
                
                audio_track_count += 1
        
        return undefined_tracks

    def find_all_audio_tracks(self, file_path: Path) -> List[Tuple[int, Dict, int, str]]:
        info = self.get_mkv_info(file_path)
        all_tracks = []
        
        if 'streams' not in info:
            return all_tracks
        
        audio_track_count = 0
        for i, stream in enumerate(info['streams']):
            if stream.get('codec_type') == 'audio':
                tags = stream.get('tags', {})
                current_language = None
                for key in tags:
                    if key.lower() in ['language', 'lang']:
                        current_language = tags[key].lower().strip()
                        break
                
                # Normalize the current language code
                current_language = self.normalize_language_code(current_language)
                
                all_tracks.append((audio_track_count, stream, i, current_language))
                if self.config.show_details:
                    lang_display = current_language if current_language else 'missing'
                    logger.info(f"Found audio track {audio_track_count} (stream {i}) in {file_path.name} - current language: '{lang_display}'")
                
                audio_track_count += 1
        
        return all_tracks

    def get_file_duration(self, file_path: Path) -> float:
        try:
            cmd = [
                self.ffprobe, '-v', 'quiet', '-show_entries', 'format=duration',
                '-of', 'csv=p=0', str(file_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding='utf-8', errors='replace')
            duration = float(result.stdout.strip())
            return duration
        except (subprocess.CalledProcessError, ValueError) as e:
            if self.config.show_details:
                logger.debug(f"Could not get duration for {file_path}: {e}")
            return 0.0
        
    def _has_reasonable_volume(self, audio_path: Path) -> bool:
        try:
            volume_cmd = [
                self.ffmpeg, '-i', str(audio_path), '-af', 'volumedetect', 
                '-f', 'null', '-', '-v', 'quiet', '-stats'
            ]
            
            result = subprocess.run(volume_cmd, capture_output=True, text=True, 
                                  encoding='utf-8', errors='replace')
            
            if 'mean_volume:' in result.stderr:
                for line in result.stderr.split('\n'):
                    if 'mean_volume:' in line:
                        try:
                            volume_db = float(line.split('mean_volume:')[1].split('dB')[0].strip())
                            # Reject if extremely quiet (likely silence)
                            return volume_db > -60.0
                        except (ValueError, IndexError):
                            pass
            
            return True  # If we can't determine, assume it's okay
        except:
            return True  # If check fails, assume it's okay
    

    
    def _attempt_transcription(self, audio_path: Path, use_vad: bool, attempt_name: str) -> Optional[Dict]:
        try:
            if self.config.show_details:
                logger.info(f"Starting transcription ({attempt_name})...")
            else:
                print(f"Transcribing audio ({attempt_name})...", flush=True)
            
            # Configure VAD options if enabled
            vad_options = None
            if use_vad:
                vad_options = {
                    "min_speech_duration_ms": self.config.vad_min_speech_duration_ms,
                    "max_speech_duration_s": self.config.vad_max_speech_duration_s
                }
            
            if attempt_name == "with_vad":
                temperature = 0.0
            else:
                # Use slightly higher temperature for non-VAD attempts to reduce repetitive hallucinations
                temperature = 0.2
            
            # Transcribe with anti-hallucination parameters
            segments, info = self.whisper_model.transcribe(
                str(audio_path),
                language=None,
                task="transcribe",
                beam_size=3,
                best_of=2,
                temperature=temperature,
                patience=1.0,
                length_penalty=1.0,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                compression_ratio_threshold=2.0,
                log_prob_threshold=-0.8,
                no_speech_threshold=0.6,
                condition_on_previous_text=False,
                initial_prompt=None,
                word_timestamps=False,
                prepend_punctuations="\"'([{-",
                append_punctuations="\"'.,:!?)]}",
                vad_filter=use_vad,
                vad_parameters=vad_options
            )
            
            if self.config.show_details:
                logger.info(f"Transcription completed, processing results...")
            else:
                print("Processing transcription results...", flush=True)
            
            # Process results
            segments_list = list(segments)
            text_sample = ' '.join([segment.text for segment in segments_list]).strip()
            
            # Track if VAD removed all audio (important for hallucination detection)
            vad_removed_all = use_vad and len(segments_list) == 0
            
            # Calculate confidence
            confidence = info.language_probability
            if segments_list:
                segment_confidences = []
                for segment in segments_list:
                    if hasattr(segment, 'avg_logprob') and segment.avg_logprob is not None:
                        segment_conf = min(1.0, max(0.0, (segment.avg_logprob + 1.0)))
                        segment_confidences.append(segment_conf)
                
                if segment_confidences:
                    avg_confidence = sum(segment_confidences) / len(segment_confidences)
                    confidence = max(confidence, avg_confidence)
            
            if self.config.show_details:
                logger.info(f"Detected language: {info.language} (confidence: {confidence:.2f}, method: {attempt_name})")
                logger.info(f"Sample text: '{text_sample[:150]}'")
                logger.info(f"Segments found: {len(segments_list)}")
            
            return {
                'language': info.language,
                'confidence': confidence,
                'text': text_sample,
                'text_length': len(text_sample),
                'word_count': len(text_sample.split()) if text_sample else 0,
                'segments_detected': len(segments_list),
                'attempt_name': attempt_name,
                'vad_removed_all': vad_removed_all
            }
            
        except Exception as e:
            if self.config.show_details:
                logger.debug(f"Transcription attempt '{attempt_name}' failed: {e}")
            return None
    
    def _process_transcription_result(self, result: Dict) -> Optional[str]:
        """Process transcription result and determine if it's valid speech."""
        
        # Skip hallucination check for very high confidence results
        if result['confidence'] > 0.95 and result['text_length'] > 50:
            language_code = self.language_codes.get(result['language'].lower(), result['language'])
            if result['language'].lower() in ['dutch', 'nl']:
                language_code = 'dut'
            return language_code
        
        if self.config.show_details:
            logger.info("Checking transcription quality and hallucination patterns...")
        else:
            print("Analyzing transcription quality...", flush=True)
        if result['text'] and self._is_likely_hallucination(result['text']):
            if self.config.show_details:
                logger.info("Detected likely hallucination - marking as 'no linguistic content' (zxx)")
            return 'zxx'
        
        # Special case: If VAD removed all audio but we still got text, be very suspicious
        if result['attempt_name'] == 'without_vad' and result.get('vad_removed_all', False):
            # Much stricter criteria when VAD removed all audio
            has_speech = (
                # Only accept very high confidence with substantial text
                (result['confidence'] > 0.7 and result['text_length'] > 30 and result['word_count'] > 5) or
                # Or extremely substantial text with decent confidence
                (result['confidence'] > 0.5 and result['text_length'] > 100 and result['word_count'] > 20)
            )
            
            if not has_speech:
                if self.config.show_details:
                    logger.info(f"VAD removed all audio but transcription produced text - likely hallucination")
                    logger.info(f"Low confidence ({result['confidence']:.3f}) with limited text - marking as 'zxx'")
                return 'zxx'
        else:
            # Normal criteria for when VAD didn't remove everything
            has_speech = (
                # High confidence with any text
                (result['confidence'] > 0.6 and result['text_length'] > 0) or
                # Medium confidence with reasonable text
                (result['confidence'] > 0.3 and result['text_length'] > 15 and result['word_count'] > 2) or
                # Lower confidence but substantial text content
                (result['confidence'] > 0.2 and result['text_length'] > 50 and result['word_count'] > 8) or
                # Lots of text regardless of confidence (likely real speech)
                (result['text_length'] > 100 and result['word_count'] > 15)
            )
        
        if has_speech:
            language_code = self.language_codes.get(result['language'].lower(), result['language'])
            
            # Handle special cases
            if result['language'].lower() in ['dutch', 'nl']:
                language_code = 'dut'
            
            # NORMALIZE the detected language code
            language_code = self.normalize_language_code(language_code)
            
            return language_code
        else:
            if self.config.show_details:
                logger.info("Insufficient evidence of speech - marking as 'no linguistic content' (zxx)")
                logger.info(f"Criteria: confidence={result['confidence']:.3f}, text_length={result['text_length']}, word_count={result['word_count']}")
            return 'zxx'
		
    def extract_audio_sample_percentage_based(self, file_path: Path, audio_track_index: int, stream_index: int, retry_attempt: int = 0) -> Optional[Path]:
        try:
            # Get file duration
            duration = self.get_file_duration(file_path)
            if duration <= 0:
                if self.config.show_details:
                    logger.warning(f"Could not determine duration for {file_path}, using fixed time samples")
                duration = 7200  # Assume 2 hours for fallback
            
            if self.config.show_details:
                logger.info(f"File duration: {duration/60:.1f} minutes")
            
            # Calculate percentage-based start times (skip first and last 5% to avoid credits/intros)
            min_start = max(60, duration * 0.05)  # At least 1 minute or 5% in
            max_start = duration * 0.85  # Don't go past 85%
            
            # Use different percentage sets for retry attempts
            if duration > 3600:  # > 1 hour
                sample_duration = 90
                all_percentages = [
                    [0.15, 0.25, 0.35, 0.50, 0.65],  # Retry 0: original samples
                    [0.08, 0.20, 0.45, 0.75, 0.88],  # Retry 1: different positions
                    [0.12, 0.40, 0.60, 0.80, 0.90],  # Retry 2: more toward end
                ]
            elif duration > 1800:  # > 30 minutes  
                sample_duration = 75
                all_percentages = [
                    [0.15, 0.30, 0.50, 0.70],        # Retry 0: original samples
                    [0.08, 0.40, 0.65, 0.85],        # Retry 1: different positions
                    [0.25, 0.45, 0.75, 0.90],        # Retry 2: more toward end
                ]
            else:
                sample_duration = 60
                all_percentages = [
                    [0.20, 0.50, 0.80],               # Retry 0: original samples
                    [0.10, 0.35, 0.75],               # Retry 1: different positions
                    [0.30, 0.60, 0.90],               # Retry 2: more toward end
                ]
            
            # Select percentage set based on retry attempt
            percentages = all_percentages[min(retry_attempt, len(all_percentages) - 1)]
            
            # Calculate actual start times
            time_segments = []
            for pct in percentages:
                start_time = max(min_start, min(max_start, duration * pct))
                time_segments.append((int(start_time), sample_duration))
            
            if self.config.show_details:
                logger.info(f"Will try {len(time_segments)} samples: " + 
                           ", ".join([f"{int(start/60)}m{start%60:02.0f}s" for start, _ in time_segments]))
            
            # Audio mapping strategies
            mapping_strategies = [
                f'0:a:{audio_track_index}',
                f'0:{stream_index}',
                f'a:{audio_track_index}',
            ]
            
            for segment_start, segment_duration in time_segments:
                for map_strategy in mapping_strategies:
                    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                    temp_path = Path(temp_file.name)
                    temp_file.close()
                    
                    try:
                        cmd = [
                            self.ffmpeg, '-y', '-v', 'error',
                            '-ss', str(segment_start),
                            '-i', str(file_path),
                            '-t', str(segment_duration),
                            '-map', map_strategy,
                            '-ar', '16000',
                            '-ac', '1',
                            '-af', 'volume=2.0,highpass=f=80,lowpass=f=8000,dynaudnorm=f=200:g=3',
                            '-f', 'wav',
                            str(temp_path)
                        ]
                        
                        limited_cmd = limit_subprocess_resources(cmd)
                        
                        if self.config.show_details:
                            logger.debug(f"Extracting from {int(segment_start/60)}m{segment_start%60:02.0f}s ({segment_duration}s)")
                        
                        result = subprocess.run(limited_cmd, check=True, capture_output=True, 
                                              text=True, encoding='utf-8', errors='replace')
                        
                        if temp_path.exists() and temp_path.stat().st_size > 10000:  # At least 10KB
                            # Quick validation - check if it has reasonable volume
                            if self._has_reasonable_volume(temp_path):
                                if self.config.show_details:
                                    logger.info(f"Successfully extracted audio from {int(segment_start/60)}m{segment_start%60:02.0f}s")
                                return temp_path
                            else:
                                if self.config.show_details:
                                    logger.debug(f"Sample from {int(segment_start/60)}m{segment_start%60:02.0f}s has very low volume")
                        
                        if temp_path.exists():
                            temp_path.unlink()
                            
                    except subprocess.CalledProcessError as e:
                        if temp_path.exists():
                            temp_path.unlink()
                        if self.config.show_details:
                            logger.debug(f"Failed to extract from {int(segment_start/60)}m{segment_start%60:02.0f}s: {e}")
                        continue
            
            logger.error("All percentage-based extraction attempts failed")
            return None
            
        except Exception as e:
            logger.error(f"Unexpected error during audio extraction: {e}")
            return None

    def extract_full_audio_track(self, file_path: Path, audio_track_index: int, stream_index: int) -> Optional[Path]:
        try:
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_path = Path(temp_file.name)
            temp_file.close()
            
            # Audio mapping strategies
            mapping_strategies = [
                f'0:a:{audio_track_index}',
                f'0:{stream_index}',
                f'a:{audio_track_index}',
            ]
            
            for map_strategy in mapping_strategies:
                try:
                    cmd = [
                        self.ffmpeg, '-y', '-v', 'error',
                        '-i', str(file_path),
                        '-map', map_strategy,
                        '-ar', '16000',
                        '-ac', '1',
                        '-af', 'volume=2.0,highpass=f=80,lowpass=f=8000,dynaudnorm=f=200:g=3',
                        '-f', 'wav',
                        str(temp_path)
                    ]
                    
                    limited_cmd = limit_subprocess_resources(cmd)
                    
                    if self.config.show_details:
                        logger.info(f"Extracting full audio track {audio_track_index}")
                    
                    result = subprocess.run(limited_cmd, check=True, capture_output=True, 
                                          text=True, encoding='utf-8', errors='replace')
                    
                    if temp_path.exists() and temp_path.stat().st_size > 10000:  # At least 10KB
                        if self.config.show_details:
                            logger.info(f"Successfully extracted full audio track {audio_track_index}")
                        return temp_path
                    
                    if temp_path.exists():
                        temp_path.unlink()
                        
                except subprocess.CalledProcessError as e:
                    if temp_path.exists():
                        temp_path.unlink()
                    if self.config.show_details:
                        logger.debug(f"Failed to extract full audio with mapping {map_strategy}: {e}")
                    continue
            
            logger.error("All full audio extraction attempts failed")
            return None
            
        except Exception as e:
            logger.error(f"Unexpected error during full audio extraction: {e}")
            return None
			
    def detect_language_with_fallback(self, audio_path: Path) -> Optional[str]:
        if not audio_path.exists() or audio_path.stat().st_size < 1000:
            logger.error(f"Audio file is too small or doesn't exist: {audio_path}")
            return None
        
        if self.config.show_details:
            logger.info(f"Analyzing audio file: {audio_path} ({audio_path.stat().st_size} bytes)")
        
        vad_removed_all_audio = False
        
        # First attempt: with VAD (if enabled in config)
        if self.config.vad_filter:
            result = self._attempt_transcription(audio_path, use_vad=True, attempt_name="with_vad")
            if result and result['segments_detected'] > 0:
                return self._process_transcription_result(result)
            elif result and result['segments_detected'] == 0:
                vad_removed_all_audio = True
                if self.config.show_details:
                    logger.info(f"VAD removed all audio, trying without VAD...")
        
        # Second attempt: without VAD
        result = self._attempt_transcription(audio_path, use_vad=False, attempt_name="without_vad")
        if result:
            # Pass information about whether VAD removed all audio
            result['vad_removed_all'] = vad_removed_all_audio
            return self._process_transcription_result(result)
        
        logger.error("Both transcription attempts failed")
        return None

    def detect_language_with_confidence(self, audio_path: Path) -> Optional[Dict]:
        if not audio_path.exists() or audio_path.stat().st_size < 1000:
            logger.error(f"Audio file is too small or doesn't exist: {audio_path}")
            return None
        
        if self.config.show_details:
            logger.info(f"Analyzing audio file: {audio_path} ({audio_path.stat().st_size} bytes)")
        
        vad_removed_all_audio = False
        
        # First attempt: with VAD (if enabled in config)
        if self.config.vad_filter:
            result = self._attempt_transcription(audio_path, use_vad=True, attempt_name="with_vad")
            if result and result['segments_detected'] > 0:
                language_code = self._process_transcription_result(result)
                return {
                    'language_code': language_code,
                    'confidence': result['confidence'],
                    'method': 'with_vad'
                }
            elif result and result['segments_detected'] == 0:
                vad_removed_all_audio = True
                if self.config.show_details:
                    logger.info(f"VAD removed all audio, trying without VAD...")
        
        # Second attempt: without VAD
        result = self._attempt_transcription(audio_path, use_vad=False, attempt_name="without_vad")
        if result:
            # Pass information about whether VAD removed all audio
            result['vad_removed_all'] = vad_removed_all_audio
            language_code = self._process_transcription_result(result)
            return {
                'language_code': language_code,
                'confidence': result['confidence'],
                'method': 'without_vad'
            }
        
        logger.error("Both transcription attempts failed")
        return None
    
    def detect_language_with_retries(self, file_path: Path, audio_track_index: int, stream_index: int, max_retries: int = 3) -> Optional[str]:
        successful_detections = []
        best_confidence = 0.0
        best_result = None
        
        for retry_attempt in range(max_retries):
            if retry_attempt > 0:
                if self.config.show_details:
                    logger.info(f"Retry attempt {retry_attempt + 1}/{max_retries} - trying different audio samples")
            
            # Extract audio sample for this retry attempt
            audio_sample = self.extract_audio_sample_percentage_based(file_path, audio_track_index, stream_index, retry_attempt)
            if not audio_sample:
                if self.config.show_details:
                    logger.warning(f"Retry {retry_attempt + 1}: Failed to extract audio sample")
                continue
            
            try:
                # Detect language for this sample and get confidence
                result_with_confidence = self.detect_language_with_confidence(audio_sample)
                
                # Clean up the temporary audio file
                if audio_sample.exists():
                    audio_sample.unlink()
                
                if result_with_confidence:
                    language_code = result_with_confidence.get('language_code')
                    confidence = result_with_confidence.get('confidence', 0.0)
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_result = language_code
                    
                    if language_code:
                        successful_detections.append(language_code)
                        
                        # If we got a real language (not zxx) with high confidence, use it immediately
                        if language_code != 'zxx' and confidence >= self.config.confidence_threshold:
                            if self.config.show_details:
                                logger.info(f"Successfully detected language '{language_code}' with confidence {confidence:.3f} on attempt {retry_attempt + 1}")
                            return language_code
                    
            except Exception as e:
                if self.config.show_details:
                    logger.warning(f"Retry {retry_attempt + 1}: Error during language detection: {e}")
                # Clean up on error
                if audio_sample and audio_sample.exists():
                    audio_sample.unlink()
                continue
        
        # Check if best confidence meets threshold
        if best_confidence >= self.config.confidence_threshold and best_result and best_result != 'zxx':
            if self.config.show_details:
                logger.info(f"Using best sample result: '{best_result}' with confidence {best_confidence:.3f}")
            return best_result
        
        # If confidence is below threshold, analyze full audio track
        if self.config.show_details:
            logger.info(f"Best confidence ({best_confidence:.3f}) below threshold ({self.config.confidence_threshold:.3f})")
            logger.info("Analyzing full audio track for better accuracy...")
        else:
            print(f"Low confidence detected, analyzing full audio track for track {audio_track_index}...")
        
        # Extract and analyze full audio track
        full_audio = self.extract_full_audio_track(file_path, audio_track_index, stream_index)
        if full_audio:
            try:
                result_with_confidence = self.detect_language_with_confidence(full_audio)
                
                # Clean up the temporary audio file
                if full_audio.exists():
                    full_audio.unlink()
                
                if result_with_confidence:
                    language_code = result_with_confidence.get('language_code')
                    confidence = result_with_confidence.get('confidence', 0.0)
                    
                    if self.config.show_details:
                        logger.info(f"Full track analysis result: '{language_code}' with confidence {confidence:.3f}")
                    
                    # Check if full track analysis meets confidence threshold
                    if language_code and language_code != 'zxx' and confidence >= self.config.confidence_threshold:
                        if self.config.show_details:
                            logger.info(f"Full track analysis confidence ({confidence:.3f}) meets threshold ({self.config.confidence_threshold:.3f})")
                        return language_code
                    elif language_code == 'zxx':
                        # Always accept zxx regardless of confidence
                        return language_code
                    else:
                        # Full track analysis didn't meet threshold
                        if self.config.show_details:
                            logger.info(f"Full track analysis confidence ({confidence:.3f}) below threshold ({self.config.confidence_threshold:.3f})")
                            logger.info("Marking as 'no linguistic content' due to insufficient confidence")
                        return 'zxx'
                        
            except Exception as e:
                logger.warning(f"Error during full track analysis: {e}")
                # Clean up on error
                if full_audio and full_audio.exists():
                    full_audio.unlink()
        
        # Fall back to sample-based results if full track analysis failed
        if successful_detections:
            # Count non-zxx detections
            real_languages = [lang for lang in successful_detections if lang != 'zxx']
            zxx_count = successful_detections.count('zxx')
            
            if real_languages:
                # If we found any real languages, use the most common one
                from collections import Counter
                most_common = Counter(real_languages).most_common(1)[0][0]
                if self.config.show_details:
                    logger.info(f"Multiple attempts found real language(s): {real_languages}, using: {most_common}")
                return most_common
            elif zxx_count == len(successful_detections):
                # All successful attempts said no speech
                if self.config.show_details:
                    logger.info(f"All {len(successful_detections)} attempts detected no linguistic content - marking as 'zxx'")
                return 'zxx'
        
        # If we get here, all attempts failed
        logger.warning(f"All language detection attempts failed for track {audio_track_index}")
        return None

    def _is_likely_hallucination(self, text: str) -> bool:
        if not text or len(text.strip()) == 0:
            return True
        
        text = text.strip()
        
        # Check for very short text that's likely noise
        if len(text) < 3:
            return True
        
        # Check for repetitive single characters or very short sequences
        if len(set(text.replace(' ', ''))) <= 3 and len(text) > 10:
            return True
        
        # Check for extremely repetitive character patterns
        unique_chars = len(set(text.replace(' ', '').replace('\n', '')))
        if unique_chars <= 2 and len(text) > 20:
            return True
        
        # Check for patterns like "ლლლლლ" or "rrrrrr" (your specific example)
        import re
        # Look for 5+ repeated characters
        if re.search(r'(.)\1{4,}', text):
            return True
        
        # Check for alternating character patterns like "abababa"
        if re.search(r'(.{1,3})\1{3,}', text):
            return True
        
        # Check for non-Latin scripts that might indicate hallucination
        non_latin_count = 0
        for char in text:
            if ord(char) > 127:  # Non-ASCII
                # Check if it's from scripts commonly hallucinated on silent audio
                if (0x1780 <= ord(char) <= 0x17FF or  # Khmer
                    0x0E00 <= ord(char) <= 0x0E7F or  # Thai
                    0x1000 <= ord(char) <= 0x109F or  # Myanmar
                    0x0980 <= ord(char) <= 0x09FF or  # Bengali
                    0x10A0 <= ord(char) <= 0x10FF):   # Georgian
                    non_latin_count += 1
        
        # If more than 70% of characters are from potentially hallucinated scripts
        if len(text) > 0 and (non_latin_count / len(text)) > 0.7:
            return True
        
        # Check for very repetitive patterns in words
        words = text.split()
        if len(words) > 3:
            unique_words = set(words)
            if len(unique_words) / len(words) < 0.2:  # Less than 20% unique words
                return True
        
        # Check for sequences of identical short strings
        if len(text) > 20:
            parts = text.split()
            if len(parts) > 5:
                unique_parts = set(parts)
                if len(unique_parts) <= 2:  # Only 1-2 unique "words" repeated
                    return True
        
        # Check compression ratio - hallucinated text often compresses very well
        try:
            import zlib
            compressed = zlib.compress(text.encode('utf-8'))
            compression_ratio = len(compressed) / len(text.encode('utf-8'))
            # If text compresses to less than 30% of original size, likely repetitive
            if compression_ratio < 0.3:
                return True
        except:
            pass  # If compression fails, skip this check
        
        common_hallucinations = [
            "okay up here we go",
            "i'm going to go get some water",
            "let's go",
            "here we go",
            "okay let's go",
            "alright let's go",
            "come on let's go",
            "okay here we go",
            "let me get some water",
            "i'm going to get some water",
            "i need to get some water",
            "hold on let me",
            "wait let me",
            "okay wait",
            "hold on",
            "one second",
            "just a second",
            "give me a second",
            "let me just",
        ]
        
        text_lower = text.lower().strip()
        
        # Remove punctuation for comparison
        import re
        clean_text = re.sub(r'[^\w\s]', '', text_lower)
        
        for phrase in common_hallucinations:
            if phrase in clean_text:
                if self.config.show_details:
                    logger.info(f"Detected common hallucination phrase: '{phrase}'")
                return True
        
        # Check for very generic/common short phrases that are often hallucinated
        if len(text_lower) < 50:  # Short text
            generic_patterns = [
                r'\b(okay|ok|alright|let\'s|here we go|come on)\b.*\b(go|water|get|just|wait)\b',
                r'\bi\'m (going to|gonna) (go|get)',
                r'\b(hold on|wait|give me|let me) (a |just |)?(second|minute|moment)\b',
            ]
            
            for pattern in generic_patterns:
                if re.search(pattern, clean_text):
                    if self.config.show_details:
                        logger.info(f"Detected generic hallucination pattern: {pattern}")
                    return True
        
        return False
    
    def update_mkv_language(self, file_path: Path, track_index: int, language_code: str, 
                          dry_run: bool = False) -> bool:
        if dry_run:
            print(f"[DRY RUN] Would update track {track_index} in {file_path.name} to language: {language_code}")
            return True
        
        try:
            cmd = [
                self.mkvpropedit, str(file_path),
                '--edit', f'track:a{track_index + 1}',
                '--set', f'language={language_code}'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            if self.config.show_details:
                logger.info(f"Updated track {track_index} in {file_path.name} to language: {language_code}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error updating {file_path}: {e}")
            return False
    
    def process_file(self, file_path: Path) -> Dict:
        results = {
            'original_file': str(file_path),
            'mkv_file': None,
            'was_remuxed': False,
            'undefined_tracks': 0,
            'processed_tracks': [],
            'failed_tracks': [],
            'errors': []
        }
        
        print(f"Processing: {file_path.name}")
        
        # Remux to MKV if needed
        mkv_path = file_path
        if self.config.remux_to_mkv and file_path.suffix.lower() != '.mkv':
            mkv_path = self.remux_to_mkv(file_path)
            if mkv_path and mkv_path != file_path:
                results['was_remuxed'] = True
                results['mkv_file'] = str(mkv_path)
            elif not mkv_path:
                results['errors'].append("Failed to remux to MKV format")
                return results
        
        results['mkv_file'] = str(mkv_path)
        
        # Find audio tracks based on config
        if self.config.reprocess_all:
            audio_tracks = self.find_all_audio_tracks(mkv_path)
            track_type = "all audio"
        else:
            # For undefined tracks, also get current language for consistency
            undefined_tracks = self.find_undefined_audio_tracks(mkv_path)
            # Convert to format with current language
            audio_tracks = []
            for audio_track_index, stream_info, stream_index in undefined_tracks:
                audio_tracks.append((audio_track_index, stream_info, stream_index, 'und'))
            track_type = "undefined audio"
        
        results['undefined_tracks'] = len(audio_tracks)  # Keep this name for compatibility
        
        if not audio_tracks:
            if self.config.show_details:
                logger.info(f"No {track_type} tracks found in {mkv_path.name}")
            return results
        
        # Process each track
        for track_data in audio_tracks:
            if len(track_data) == 4:
                audio_track_index, stream_info, stream_index, current_language = track_data
            else:
                audio_track_index, stream_info, stream_index = track_data
                current_language = 'und'
                
            try:
                # Detect language with retries
                language_code = self.detect_language_with_retries(mkv_path, audio_track_index, stream_index, max_retries=3)
                if not language_code:
                    results['failed_tracks'].append(audio_track_index)
                    results['errors'].append(f"Failed to detect language for track {audio_track_index}")
                    continue
                
                # Update metadata
                success = self.update_mkv_language(mkv_path, audio_track_index, language_code, self.config.dry_run)
                if success:
                    results['processed_tracks'].append({
                        'track_index': audio_track_index,
                        'detected_language': language_code,
                        'previous_language': current_language
                    })
                else:
                    results['failed_tracks'].append(audio_track_index)
                    results['errors'].append(f"Failed to update track {audio_track_index}")
                    
            except Exception as e:
                error_msg = f"Error processing track {audio_track_index}: {str(e)}"
                logger.error(error_msg)
                results['failed_tracks'].append(audio_track_index)
                results['errors'].append(error_msg)
        
        return results
    
    def process_directory(self, directory: str) -> List[Dict]:
        video_files = self.find_video_files(directory)
        results = []
        
        for file_path in video_files:
            try:
                result = self.process_file(file_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                results.append({
                    'original_file': str(file_path),
                    'mkv_file': None,
                    'was_remuxed': False,
                    'undefined_tracks': 0,
                    'processed_tracks': [],
                    'errors': [f"Processing failed: {str(e)}"]
                })
        
        return results

def check_for_updates():
    try:
        print("Checking for updates...", end=" ", flush=True)
        
        api_url = "https://api.github.com/repos/netplexflix/MKV-Undefined-Audio-Language-Detector/releases/latest"
        
        response = requests.get(api_url, timeout=5)
        response.raise_for_status()
        
        release_data = response.json()
        latest_version = release_data.get('tag_name', '').lstrip('v')
        
        if not latest_version:
            print("Could not determine latest version")
            return
        
        current_version = VERSION
        
        try:
            if version.parse(latest_version) > version.parse(current_version):
                print("UPDATE AVAILABLE!")
                print(f"\n{'='*60}")
                print("📄 UPDATE AVAILABLE")
                print(f"{'='*60}")
                print(f"Current version: {current_version}")
                print(f"Latest version:  {latest_version}")
                print("Download from: https://github.com/netplexflix/MKV-Undefined-Audio-Language-Detector")
                print(f"{'='*60}\n")
            else:
                print(f"✓ Up to date. Version: {VERSION}")
                
        except Exception as e:
            if latest_version != current_version:
                print("Update may be available")
                print(f"Current: {current_version}, Latest: {latest_version}")
                print("Check: https://github.com/netplexflix/MKV-Undefined-Audio-Language-Detector\n")
            else:
                print("f✓ Up to date. Version: {VERSION}")
        
    except requests.exceptions.RequestException as e:
        print("Failed (network error)")
        if logger.isEnabledFor(logging.INFO):
            logger.info(f"Update check failed: {e}")
    except Exception as e:
        print("Failed (error)")
        if logger.isEnabledFor(logging.INFO):
            logger.info(f"Update check error: {e}")

def format_duration(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def print_detailed_summary(results: List[Dict], config: Config, runtime_seconds: float, detector: MKVLanguageDetector = None):
    print(f"\n{'='*60}")
    print("PROCESSING SUMMARY")
    print(f"{'='*60}")
    
    total_files = len(results)
    files_with_actions = 0
    files_successfully_processed = 0
    files_with_failures = 0
    files_processed = 0
    silent_content_files = []
    
    # ANSI color codes
    YELLOW = '\033[93m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    
    for result in results:
        file_name = Path(result['original_file']).name
        actions = []
        has_silent_content = False
        file_has_failures = False
        
        if result['was_remuxed']:
            actions.append("remuxed")
        
        processed_tracks = sorted(result['processed_tracks'], key=lambda x: x['track_index'])
        failed_tracks = sorted(result.get('failed_tracks', []))

        if failed_tracks or any(error for error in result['errors'] if not any(track_phrase in error for track_phrase in 
                               ['Failed to extract audio from track', 'Failed to detect language for track', 'Failed to update track'])):
            file_has_failures = True
            
        track_actions = []
        
        for track in processed_tracks:
            track_idx = track['track_index']
            lang_code = track['detected_language']
            prev_lang = track.get('previous_language', 'und')
            
            # Format the track action based on whether language changed
            if prev_lang == lang_code:
                # No change - show in normal color
                if lang_code == 'zxx':
                    track_actions.append(f"{YELLOW}track{track_idx}: {prev_lang} -> {lang_code} (no speech){RESET}")
                    has_silent_content = True
                else:
                    track_actions.append(f"track{track_idx}: {prev_lang} -> {lang_code}")
            else:
                # Language changed - show in color
                if lang_code == 'zxx':
                    track_actions.append(f"{YELLOW}track{track_idx}: {prev_lang} -> {lang_code} (no speech){RESET}")
                    has_silent_content = True
                else:
                    track_actions.append(f"{CYAN}track{track_idx}: {prev_lang} -> {lang_code}{RESET}")
        
        for track_idx in failed_tracks:
            track_actions.append(f"{RED}track{track_idx}: failed{RESET}")
        
        major_errors = [e for e in result['errors'] if not any(track_phrase in e for track_phrase in 
                       ['Failed to extract audio from track', 'Failed to detect language for track', 'Failed to update track'])]
        
        show_file = False
        
        if actions or track_actions:
            all_actions = actions + track_actions
            print(f"{file_name}: {', '.join(all_actions)}")
            files_processed += 1
            show_file = True
            if has_silent_content:
                silent_content_files.append(file_name)
        elif major_errors:
            print(f"{file_name}: {RED}error{RESET} - {major_errors[0]}")
            show_file = True
            file_has_failures = True
        
        if show_file:
            files_with_actions += 1
            if file_has_failures:
                files_with_failures += 1
            else:
                files_successfully_processed += 1
    
    if files_with_actions > 0:
        print(f"\nShowing {files_with_actions} files that required action (out of {total_files} total files)")
        
        status_parts = []
        if files_successfully_processed > 0:
            status_parts.append(f"Successfully processed {files_successfully_processed} files")
        if files_with_failures > 0:
            status_parts.append(f"{RED}{files_with_failures} files failed!{RESET}")
        
        if status_parts:
            print(". ".join(status_parts))
    else:
        print("No files required any action")
    
    # Special warning for silent content
    if silent_content_files:
        print(f"\n{YELLOW}⚠️  WARNING: Silent content detected in {len(silent_content_files)} file(s){RESET}")
        print(f"{YELLOW}   These tracks were marked as 'zxx' (no linguistic content). You may want to manually verify them.{RESET}")
        if len(silent_content_files) <= 5:
            for filename in silent_content_files:
                print(f"{YELLOW}   - {filename}{RESET}")
        else:
            for filename in silent_content_files[:3]:
                print(f"{YELLOW}   - {filename}{RESET}")
            print(f"{YELLOW}   ... and {len(silent_content_files) - 3} more{RESET}")
    
    # Show deletion failures if any occurred
    if detector and hasattr(detector, 'deletion_failures') and detector.deletion_failures:
        print(f"\n{YELLOW}⚠️  WARNING: {len(detector.deletion_failures)} original file(s) could not be deleted after remuxing. (read-only?){RESET}")
        
        for failure in detector.deletion_failures:
            original_name = Path(failure['original_file']).name
            print(f"{YELLOW}   - {original_name}{RESET}")
            if config.show_details:
                print(f"{YELLOW}     Error: {failure['error']}{RESET}")
                print(f"{YELLOW}     Original: {failure['original_file']}{RESET}")
                print(f"{YELLOW}     MKV: {failure['mkv_file']}{RESET}")
    
    print(f"\nTotal runtime: {format_duration(runtime_seconds)}")
    
    if config.dry_run:
        print("(Dry run - no files were actually modified)")
    print()

def main():
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description='Detect and update language metadata for video file audio tracks')
    parser.add_argument('--config', default='config/config.yml', 
                       help='Configuration file path (default: config/config.yml)')
    parser.add_argument('--create-config', action='store_true',
                       help='Create a sample configuration file')
    parser.add_argument('--directory', help='Override directory from config')
    parser.add_argument('--model', choices=['tiny', 'base', 'small', 'medium', 'large'], 
                       help='Override Whisper model size from config')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Preview changes without modifying files')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Force verbose logging (overrides config)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Force quiet mode (overrides config)')
    parser.add_argument('--find-mkv', action='store_true',
                       help='Help locate MKVToolNix installation')
    parser.add_argument('--skip-update-check', action='store_true',
                       help='Skip checking for updates on GitHub')
    parser.add_argument('--no-vad', action='store_true',
                       help='Disable VAD filter (overrides config)')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'],
                       help='Override device selection from config')
    parser.add_argument('--compute-type', 
                       choices=['auto', 'int8', 'int8_float16', 'int16', 'float16', 'float32'],
                       help='Override compute type from config')
    parser.add_argument('--reprocess-all', action='store_true',
                       help='Reprocess all audio tracks instead of only undefined ones')
    
    args = parser.parse_args()
    
    # Handle config file creation
    if args.create_config:
        config = Config()
        config.create_sample_config(args.config)
        return
    
    # Handle MKVToolNix location help
    if args.find_mkv:
        if sys.platform == 'win32':
            found_path = find_mkvtoolnix_installation()
            if found_path:
                print(f"\nTo fix the PATH issue, add this directory to your PATH environment variable:")
                print(f"{os.path.dirname(found_path)}")
                print(f"\nOr copy this path to use directly: {found_path}")
            else:
                print("MKVToolNix not found. Please install it from:")
                print("https://mkvtoolnix.download/downloads.html")
        else:
            print("This option is only available on Windows")
        return
    
    # Check for updates first
    if not args.skip_update_check:
        check_for_updates()
    else:
        print("Skipping update check")
    
    # Load configuration
    config = Config()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
        config.show_details = True
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
        config.show_details = False
    else:
        logging.getLogger().setLevel(logging.WARNING)
    
    config.load_from_file(args.config)
    
    # Override config with command line arguments
    if args.directory:
        config.path = args.directory
    if args.model:
        config.whisper_model = args.model
    if args.dry_run:
        config.dry_run = True
    if args.no_vad:
        config.vad_filter = False
    if args.device:
        config.device = args.device
    if args.compute_type:
        config.compute_type = args.compute_type
    if args.reprocess_all:
        config.reprocess_all = True
    
    # Set final logging level based on configuration (after loading config)
    if config.show_details:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)
       
    # Validate directory
    for d in config.path:
        if not os.path.isdir(d):
            logger.error(f"Directory not found: {d}")
            sys.exit(1)

    
    # Check dependencies
    dependencies = {
        'ffmpeg': 'FFmpeg (https://ffmpeg.org/download.html)',
        'ffprobe': 'FFmpeg (https://ffmpeg.org/download.html)', 
        'mkvpropedit': 'MKVToolNix (https://mkvtoolnix.download/downloads.html)'
    }
    
    missing_deps = []
    for dep, source in dependencies.items():
        if not find_executable(dep):
            missing_deps.append((dep, source))
    
    if missing_deps:
        logger.error("Missing required dependencies:")
        for dep, source in missing_deps:
            logger.error(f"  - {dep}: Install from {source}")
        
        if sys.platform == 'win32':
            logger.error("\nWindows installation options:")
            logger.error("1. Download installers from the URLs above")
            logger.error("2. Use Chocolatey: choco install ffmpeg mkvtoolnix")
            logger.error("3. Use winget: winget install FFmpeg && winget install MKVToolNix.MKVToolNix")
            logger.error("4. Make sure executables are in your PATH environment variable")
            
            if 'mkvpropedit' in [dep for dep, _ in missing_deps]:
                logger.error("\nTo help locate MKVToolNix, run:")
                logger.error("python MUALD.py --find-mkv")
        
        sys.exit(1)
    
    # Check for faster-whisper dependency
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        logger.error("faster-whisper is not installed. Install it with:")
        logger.error("pip install faster-whisper")
        logger.error("\nFor CUDA GPU support, also install:")
        logger.error("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        sys.exit(1)
    
    # Initialize detector
    try:
        if config.show_details:
            logger.info(f"Loading faster-whisper model: {config.whisper_model}")
            if config.vad_filter:
                logger.info(f"VAD filter enabled (min_speech: {config.vad_min_speech_duration_ms}ms, max_speech: {config.vad_max_speech_duration_s}s)")
            else:
                logger.info("VAD filter disabled")
        else:
            print(f"Loading faster-whisper model: {config.whisper_model}")
            if config.vad_filter:
                print("VAD filter enabled")
        
        detector = MKVLanguageDetector(config)
    except RuntimeError as e:
        logger.error(f"Failed to initialize detector: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error initializing faster-whisper: {e}")
        logger.error("Try installing with: pip install faster-whisper")
        sys.exit(1)
    
    # Process directory
    if config.show_details:
        logger.info(f"Scanning directory: {config.path}")
    else:
        print(f"Scanning directory: {config.path}")
        
    if config.dry_run:
        print("DRY RUN MODE - No files will be modified")
    
    results = []
    for d in config.path:
        if config.show_details:
            logger.info(f"Scanning directory: {d}")
        else:
            print(f"Scanning directory: {d}")
        results.extend(detector.process_directory(d))
    
    end_time = time.time()
    runtime_seconds = end_time - start_time
    
    print_detailed_summary(results, config, runtime_seconds, detector)

if __name__ == "__main__":
    main()