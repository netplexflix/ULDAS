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
import re

VERSION = '2025.11.27'

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
        self.device = "auto"
        self.compute_type = "auto"
        self.cpu_threads = 0
        self.confidence_threshold = 0.9
        self.reprocess_all = False
        self.use_tracking = True
        self.force_reprocess = False
        
        # Subtitle processing options
        self.process_subtitles = False
        self.analyze_forced_subtitles = False
        self.detect_sdh_subtitles = True
        self.subtitle_confidence_threshold = 0.85
        self.reprocess_all_subtitles = False
        
        # Timeout settings
        self.operation_timeout_seconds = 600  # 10 minutes default
        
        # Multi-factor forced subtitle detection thresholds
        # Coverage-based (secondary factor)
        self.forced_subtitle_low_coverage_threshold = 25.0   # Below = likely forced
        self.forced_subtitle_high_coverage_threshold = 50.0  # Above = likely full
        
        # Density-based (primary factor) - subtitles per minute
        self.forced_subtitle_low_density_threshold = 3.0     # Below = likely forced
        self.forced_subtitle_high_density_threshold = 8.0    # Above = likely full
        
        # Absolute count thresholds (tertiary factor)
        self.forced_subtitle_min_count_threshold = 50        # Below = likely forced
        self.forced_subtitle_max_count_threshold = 300        # Above = likely full
        
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
                
                # Load subtitle processing options
                self.process_subtitles = config_data.get('process_subtitles', self.process_subtitles)
                self.analyze_forced_subtitles = config_data.get('analyze_forced_subtitles', self.analyze_forced_subtitles)
                self.detect_sdh_subtitles = config_data.get('detect_sdh_subtitles', self.detect_sdh_subtitles)
                self.subtitle_confidence_threshold = config_data.get('subtitle_confidence_threshold', self.subtitle_confidence_threshold)
                self.reprocess_all_subtitles = config_data.get('reprocess_all_subtitles', self.reprocess_all_subtitles)
                
                # Load timeout and threshold settings
                self.operation_timeout_seconds = config_data.get('operation_timeout_seconds', self.operation_timeout_seconds)
                self.forced_subtitle_low_coverage_threshold = config_data.get('forced_subtitle_low_coverage_threshold', self.forced_subtitle_low_coverage_threshold)
                self.forced_subtitle_high_coverage_threshold = config_data.get('forced_subtitle_high_coverage_threshold', self.forced_subtitle_high_coverage_threshold)
                self.forced_subtitle_low_density_threshold = config_data.get('forced_subtitle_low_density_threshold', self.forced_subtitle_low_density_threshold)
                self.forced_subtitle_high_density_threshold = config_data.get('forced_subtitle_high_density_threshold', self.forced_subtitle_high_density_threshold)
                self.forced_subtitle_min_count_threshold = config_data.get('forced_subtitle_min_count_threshold', self.forced_subtitle_min_count_threshold)
                self.forced_subtitle_max_count_threshold = config_data.get('forced_subtitle_max_count_threshold', self.forced_subtitle_max_count_threshold)
                
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
            'reprocess_all': False,
            'process_subtitles': True,
            'analyze_forced_subtitles': True,
            'detect_sdh_subtitles': True,
            'subtitle_confidence_threshold': 0.85,
            'reprocess_all_subtitles': False,
            
            # Timeout settings
            'operation_timeout_seconds': 600,  # 10 minutes

            # Forced subtitle detection thresholds
            # Density-based (primary factor) - subtitles per minute
            'forced_subtitle_low_density_threshold': 3.0,     # Below = likely forced
            'forced_subtitle_high_density_threshold': 8.0,    # Above = likely full

            # Coverage-based (secondary factor)
            'forced_subtitle_low_coverage_threshold': 25.0,   # Below = likely forced
            'forced_subtitle_high_coverage_threshold': 50.0,  # Above = likely full
            
            # Absolute count thresholds (tertiary factor)
            'forced_subtitle_min_count_threshold': 50,        # Below = likely forced
            'forced_subtitle_max_count_threshold': 300,       # Above = likely full
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

class ProcessingTracker:
    """Tracks successfully processed files to avoid reprocessing."""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.tracker_file = self.config_dir / "processed_files.json"
        self.data = self._load()
    
    def _load(self) -> Dict:
        """Load tracking data from file."""
        if self.tracker_file.exists():
            try:
                with open(self.tracker_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not load tracking file: {e}. Starting fresh.")
                return {}
        return {}
    
    def _save(self):
        """Save tracking data to file."""
        try:
            with open(self.tracker_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2)
        except IOError as e:
            logger.error(f"Could not save tracking file: {e}")
    
    def is_processed(self, file_path: Path) -> bool:
        """Check if file has been successfully processed."""
        key = str(file_path.absolute())
        
        if key not in self.data:
            return False
        
        entry = self.data[key]
        
        # Verify file still exists and hasn't changed
        if not file_path.exists():
            # File was deleted, remove from tracking
            del self.data[key]
            self._save()
            return False
        
        stat = file_path.stat()
        current_size = stat.st_size
        current_mtime = stat.st_mtime
        
        # Check if size or modification time changed
        if (entry['size'] != current_size or 
            abs(entry['mtime'] - current_mtime) > 1):  # 1 second tolerance for filesystem quirks
            # File changed, remove from tracking
            del self.data[key]
            self._save()
            return False
        
        return True
    
    def mark_processed(self, file_path: Path, audio_success: bool = False, 
                      subtitle_success: bool = False):
        """Mark file as successfully processed."""
        # Only mark if at least one type was successfully processed
        if not (audio_success or subtitle_success):
            return
        
        key = str(file_path.absolute())
        stat = file_path.stat()
        
        self.data[key] = {
            'size': stat.st_size,
            'mtime': stat.st_mtime,
            'audio_processed': audio_success,
            'subtitle_processed': subtitle_success,
            'processed_date': time.time()
        }
        self._save()
    
    def clear_entry(self, file_path: Path):
        """Remove a file from tracking (for force reprocessing)."""
        key = str(file_path.absolute())
        if key in self.data:
            del self.data[key]
            self._save()
    
    def clear_all(self):
        """Clear all tracking data."""
        self.data = {}
        self._save()
    
    def get_stats(self) -> Dict:
        """Get statistics about tracked files."""
        total = len(self.data)
        audio_only = sum(1 for e in self.data.values() if e['audio_processed'] and not e['subtitle_processed'])
        subtitle_only = sum(1 for e in self.data.values() if e['subtitle_processed'] and not e['audio_processed'])
        both = sum(1 for e in self.data.values() if e['audio_processed'] and e['subtitle_processed'])
        
        return {
            'total_tracked': total,
            'audio_only': audio_only,
            'subtitle_only': subtitle_only,
            'both': both
        }

class MKVLanguageDetector:
    def __init__(self, config: Config):
        setup_cpu_limits()
        
        self.config = config
        
        # Initialize tracking
        if config.use_tracking:
            self.tracker = ProcessingTracker("config")
            if config.force_reprocess:
                logger.info("Force reprocess enabled - ignoring tracking cache")
        
        # Determine device and compute type
        device = self._determine_device()
        compute_type = self._determine_compute_type(device)
        cpu_threads = config.cpu_threads if config.cpu_threads > 0 else 0
        
        if config.show_details:
            logger.info(f"Initializing faster-whisper with device: {device}, compute_type: {compute_type}")
        
        # Initialize faster-whisper model
        is_local_path = os.path.isdir(config.whisper_model) or os.path.isfile(config.whisper_model)
        try:
            self.whisper_model = WhisperModel(
                config.whisper_model, 
                device=device, 
                compute_type=compute_type,
                cpu_threads=cpu_threads,
                download_root=None if not is_local_path else os.path.dirname(config.whisper_model),
                local_files_only=is_local_path
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

    def find_undefined_subtitle_tracks(self, file_path: Path) -> List[Tuple[int, Dict, int]]:
        """Find subtitle tracks with undefined language tags."""
        info = self.get_mkv_info(file_path)
        undefined_tracks = []
        
        if 'streams' not in info:
            return undefined_tracks
        
        subtitle_track_count = 0
        for i, stream in enumerate(info['streams']):
            if stream.get('codec_type') == 'subtitle':
                tags = stream.get('tags', {})
                
                language = None
                for key in tags:
                    if key.lower() in ['language', 'lang']:
                        language = tags[key].lower().strip()
                        break
                
                undefined_indicators = ['und', 'unknown', 'undefined', 'undetermined', '']
                
                if not language or language in undefined_indicators:
                    undefined_tracks.append((subtitle_track_count, stream, i))
                    if self.config.show_details:
                        lang_display = language if language else 'missing'
                        logger.info(f"Found undefined subtitle track {subtitle_track_count} (stream {i}) in {file_path.name} - language: '{lang_display}'")
                else:
                    if self.config.show_details:
                        logger.info(f"Subtitle track {subtitle_track_count} (stream {i}) already has language: '{language}' - skipping")
                
                subtitle_track_count += 1
        
        return undefined_tracks

    def find_all_subtitle_tracks(self, file_path: Path) -> List[Tuple[int, Dict, int, str]]:
        """Find all subtitle tracks regardless of language status."""
        info = self.get_mkv_info(file_path)
        all_tracks = []
        
        if 'streams' not in info:
            return all_tracks
        
        subtitle_track_count = 0
        for i, stream in enumerate(info['streams']):
            if stream.get('codec_type') == 'subtitle':
                tags = stream.get('tags', {})
                current_language = None
                for key in tags:
                    if key.lower() in ['language', 'lang']:
                        current_language = tags[key].lower().strip()
                        break
                
                current_language = self.normalize_language_code(current_language)
                
                all_tracks.append((subtitle_track_count, stream, i, current_language))
                if self.config.show_details:
                    lang_display = current_language if current_language else 'missing'
                    logger.info(f"Found subtitle track {subtitle_track_count} (stream {i}) in {file_path.name} - current language: '{lang_display}'")
                
                subtitle_track_count += 1
        
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
                        logger.info(f"Extracting full audio track {audio_track_index} (timeout: {self.config.operation_timeout_seconds}s)")
                    
                    # Add timeout to subprocess
                    result = subprocess.run(
                        limited_cmd, 
                        check=True, 
                        capture_output=True,
                        text=True, 
                        encoding='utf-8', 
                        errors='replace',
                        timeout=self.config.operation_timeout_seconds
                    )
                    
                    if temp_path.exists() and temp_path.stat().st_size > 10000:  # At least 10KB
                        if self.config.show_details:
                            logger.info(f"Successfully extracted full audio track {audio_track_index}")
                        return temp_path
                    
                    if temp_path.exists():
                        temp_path.unlink()
                        
                except subprocess.TimeoutExpired:
                    logger.error(f"Full audio extraction timed out after {self.config.operation_timeout_seconds} seconds")
                    logger.warning("Skipping audio analysis for this subtitle track")
                    if temp_path.exists():
                        temp_path.unlink()
                    return None
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

    def extract_subtitle_track(self, file_path: Path, subtitle_track_index: int, stream_index: int) -> Optional[Path]:
        """Extract subtitle track to appropriate format for analysis."""
        try:
            # First, get codec information for this subtitle track
            info = self.get_mkv_info(file_path)
            subtitle_codec = None
            
            if 'streams' in info:
                subtitle_count = 0
                for stream in info['streams']:
                    if stream.get('codec_type') == 'subtitle':
                        if subtitle_count == subtitle_track_index:
                            subtitle_codec = stream.get('codec_name', '').lower()
                            if self.config.show_details:
                                logger.info(f"Found subtitle track {subtitle_track_index}: codec='{subtitle_codec}', stream_index={stream.get('index', 'unknown')}")
                            break
                        subtitle_count += 1
            
            if self.config.show_details:
                print(f"Subtitle codec for track {subtitle_track_index}: {subtitle_codec}")
            
            # Determine output format based on codec
            # Image-based subtitles (PGS, VobSub, DVD) cannot be converted to text
            image_based_codecs = ['hdmv_pgs_subtitle', 'pgs', 'dvdsub', 'dvbsub', 'dvd_subtitle', 's_hdmv/pgs']
            text_based_codecs = ['subrip', 'srt', 'ass', 'ssa', 'webvtt', 'mov_text', 's_text/utf8', 's_text/ass']
            
            # Check if codec name contains indicators
            is_image_based = (subtitle_codec in image_based_codecs or 
                             'pgs' in subtitle_codec or 
                             'hdmv' in subtitle_codec or
                             'dvd' in subtitle_codec)
            
            if is_image_based:
                # For image-based subtitles, extract without conversion
                temp_file = tempfile.NamedTemporaryFile(suffix='.sup', delete=False)
                temp_path = Path(temp_file.name)
                temp_file.close()
                copy_codec = 'copy'
                if self.config.show_details:
                    print(f"Detected image-based subtitle (will extract as .sup)")
            else:
                # For text-based subtitles, convert to SRT
                temp_file = tempfile.NamedTemporaryFile(suffix='.srt', delete=False)
                temp_path = Path(temp_file.name)
                temp_file.close()
                copy_codec = 'srt'
                if self.config.show_details:
                    print(f"Detected text-based subtitle (will convert to .srt)")
            
            # Mapping strategies for subtitle extraction
            mapping_strategies = [
                f'0:s:{subtitle_track_index}',
                f'0:{stream_index}',
                f's:{subtitle_track_index}',
            ]
            
            for map_strategy in mapping_strategies:
                try:
                    cmd = [
                        self.ffmpeg, '-y', '-v', 'warning',
                        '-i', str(file_path),
                        '-map', map_strategy,
                        '-c:s', copy_codec,
                        str(temp_path)
                    ]
                    
                    limited_cmd = limit_subprocess_resources(cmd)
                    
                    if self.config.show_details:
                        print(f"Trying extraction with mapping: {map_strategy}")
                    
                    result = subprocess.run(limited_cmd, check=True, capture_output=True, 
                                          text=True, encoding='utf-8', errors='replace')
                    
                    if temp_path.exists() and temp_path.stat().st_size > 100:
                        if self.config.show_details:
                            print(f"✓ Successfully extracted subtitle track {subtitle_track_index} ({temp_path.stat().st_size} bytes)")
                        return temp_path
                    else:
                        if self.config.show_details:
                            print(f"✗ File too small or doesn't exist: {temp_path.stat().st_size if temp_path.exists() else 0} bytes")
                    
                    if temp_path.exists():
                        temp_path.unlink()
                        
                except subprocess.CalledProcessError as e:
                    if self.config.show_details:
                        print(f"✗ Failed with mapping {map_strategy}")
                        if e.stderr:
                            print(f"  Error: {e.stderr}")
                    if temp_path.exists():
                        temp_path.unlink()
                    continue
            
            logger.error("All subtitle extraction attempts failed")
            return None
            
        except Exception as e:
            logger.error(f"Unexpected error during subtitle extraction: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _calculate_subtitle_statistics(self, subtitles: List[Dict], duration: float) -> Dict:
        """
        Calculate comprehensive statistics about subtitle track for forced detection.
        
        Returns dict with:
            - total_duration: Total time subtitles are displayed (seconds)
            - coverage_percent: Percentage of video covered by subtitles
            - density: Subtitles per minute
            - count: Total number of subtitles
            - avg_duration: Average subtitle display time (seconds)
            - gap_variance: Variance in gaps between subtitles (high = clustered)
        """
        if not subtitles or duration <= 0:
            return {
                'total_duration': 0.0,
                'coverage_percent': 0.0,
                'density': 0.0,
                'count': 0,
                'avg_duration': 0.0,
                'gap_variance': 0.0
            }
        
        total_subtitle_duration = 0.0
        subtitle_timings = []
        subtitle_durations = []
        
        for sub in subtitles:
            try:
                start_seconds = self._parse_srt_time(sub['start'])
                end_seconds = self._parse_srt_time(sub['end'])
                subtitle_duration = end_seconds - start_seconds
                
                if subtitle_duration > 0:
                    total_subtitle_duration += subtitle_duration
                    subtitle_timings.append((start_seconds, end_seconds))
                    subtitle_durations.append(subtitle_duration)
            except:
                continue
        
        # Calculate basic metrics
        coverage_percent = (total_subtitle_duration / duration) * 100
        density = len(subtitles) / (duration / 60)  # Subtitles per minute
        avg_duration = sum(subtitle_durations) / len(subtitle_durations) if subtitle_durations else 0.0
        
        # Calculate gap variance (distribution pattern)
        gaps = []
        for i in range(len(subtitle_timings) - 1):
            gap = subtitle_timings[i + 1][0] - subtitle_timings[i][1]
            if gap >= 0:  # Only positive gaps
                gaps.append(gap)
        
        gap_variance = 0.0
        if len(gaps) > 1:
            mean_gap = sum(gaps) / len(gaps)
            gap_variance = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
        
        return {
            'total_duration': total_subtitle_duration,
            'coverage_percent': coverage_percent,
            'density': density,
            'count': len(subtitles),
            'avg_duration': avg_duration,
            'gap_variance': gap_variance,
            'subtitle_timings': subtitle_timings  # For audio analysis if needed
        }

    def _decide_forced_from_statistics(self, stats: Dict, duration_minutes: float) -> Tuple[bool, str, int]:
        """
        Decide if subtitles are forced based on multi-factor analysis.
        
        Returns:
            (is_forced, reason, confidence_level)
            confidence_level: 3=high, 2=medium, 1=low (needs audio analysis)
        """
        density = stats['density']
        coverage = stats['coverage_percent']
        count = stats['count']
        
        # TIER 1: High Confidence Decisions (no audio needed)
        
        # Definitely FORCED
        if density < self.config.forced_subtitle_low_density_threshold and coverage < self.config.forced_subtitle_low_coverage_threshold:
            return True, f"Very low density ({density:.1f} subs/min) and coverage ({coverage:.1f}%)", 3
        
        if count < self.config.forced_subtitle_min_count_threshold:
            return True, f"Very low subtitle count ({count} subtitles)", 3
        
        if density < 2.0:  # Extremely sparse
            return True, f"Extremely low density ({density:.1f} subs/min)", 3
        
        # Definitely FULL
        if density > self.config.forced_subtitle_high_density_threshold and coverage > 30.0:
            return False, f"High density ({density:.1f} subs/min) and coverage ({coverage:.1f}%)", 3
        
        if count > self.config.forced_subtitle_max_count_threshold and duration_minutes > 30:
            return False, f"High subtitle count ({count} subtitles for {duration_minutes:.0f} min video)", 3
        
        if density > 10.0:  # Very dense
            return False, f"Very high density ({density:.1f} subs/min)", 3
        
        # TIER 2: Medium Confidence Decisions
        
        forced_indicators = 0
        full_indicators = 0
        
        # Check multiple factors
        if density < 5.0:
            forced_indicators += 1
        if density > 6.0:
            full_indicators += 1
        
        if coverage < 30.0:
            forced_indicators += 1
        if coverage > 40.0:
            full_indicators += 1
        
        if count < 150:
            forced_indicators += 1
        if count > 250:
            full_indicators += 1
        
        # High gap variance suggests clustering (forced)
        if stats['gap_variance'] > 100.0:
            forced_indicators += 1
        # Low gap variance suggests even distribution (full)
        if stats['gap_variance'] < 50.0:
            full_indicators += 1
        
        if forced_indicators >= 2 and full_indicators == 0:
            factors = []
            if density < 5.0:
                factors.append(f"density={density:.1f}")
            if coverage < 30.0:
                factors.append(f"coverage={coverage:.1f}%")
            if count < 150:
                factors.append(f"count={count}")
            return True, f"Multiple forced indicators: {', '.join(factors)}", 2
        
        if full_indicators >= 2 and forced_indicators == 0:
            factors = []
            if density > 6.0:
                factors.append(f"density={density:.1f}")
            if coverage > 40.0:
                factors.append(f"coverage={coverage:.1f}%")
            if count > 250:
                factors.append(f"count={count}")
            return False, f"Multiple full indicators: {', '.join(factors)}", 2
        
        # TIER 3: Low Confidence (ambiguous - needs audio analysis)
        reason = f"Ambiguous metrics: density={density:.1f} subs/min, coverage={coverage:.1f}%, count={count}"
        
        # If audio analysis is disabled, make best guess
        if not self.config.analyze_forced_subtitles:
            # Use midpoint heuristic
            is_forced = density < 5.5 or coverage < 37.5
            return is_forced, reason + " (audio analysis disabled, using heuristic)", 1
        
        # Return low confidence - caller should do audio analysis
        return None, reason, 1
    
    def parse_srt_file(self, srt_path: Path) -> List[Dict]:
        """Parse SRT subtitle file and return list of subtitle entries."""
        try:
            with open(srt_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            # Split by double newlines to separate subtitle blocks
            blocks = content.strip().split('\n\n')
            subtitles = []
            
            for block in blocks:
                lines = block.strip().split('\n')
                if len(lines) >= 3:
                    try:
                        # Parse subtitle entry
                        index = int(lines[0].strip())
                        timing = lines[1].strip()
                        text = '\n'.join(lines[2:])
                        
                        # Parse timing
                        if ' --> ' in timing:
                            start_time, end_time = timing.split(' --> ')
                            
                            subtitles.append({
                                'index': index,
                                'start': start_time.strip(),
                                'end': end_time.strip(),
                                'text': text.strip()
                            })
                    except (ValueError, IndexError):
                        continue
            
            return subtitles
            
        except Exception as e:
            logger.error(f"Error parsing SRT file: {e}")
            return []

    def get_subtitle_text_sample(self, subtitles: List[Dict], max_chars: int = 5000) -> str:
        """Extract text sample from subtitles for language detection."""
        text_parts = []
        total_chars = 0
        
        # Sample from beginning, middle, and end
        sample_indices = []
        if len(subtitles) > 0:
            sample_indices.append(0)  # Beginning
        if len(subtitles) > 10:
            sample_indices.append(len(subtitles) // 2)  # Middle
        if len(subtitles) > 20:
            sample_indices.append(len(subtitles) - 1)  # End
        
        for idx in sample_indices:
            if idx < len(subtitles):
                # Get a chunk of subtitles around this index
                start_idx = max(0, idx - 5)
                end_idx = min(len(subtitles), idx + 5)
                
                for sub in subtitles[start_idx:end_idx]:
                    text = sub['text']
                    # Remove SRT formatting tags
                    text = re.sub(r'<[^>]+>', '', text)
                    text = re.sub(r'\{[^}]+\}', '', text)
                    text_parts.append(text)
                    total_chars += len(text)
                    
                    if total_chars >= max_chars:
                        break
            
            if total_chars >= max_chars:
                break
        
        return ' '.join(text_parts)

    def _convert_iso639_1_to_2(self, code: str) -> str:
        """Convert ISO 639-1 (2-letter) to ISO 639-2 (3-letter) code."""
        iso639_1_to_2_mapping = {
            'en': 'eng', 'es': 'spa', 'fr': 'fre', 'de': 'ger', 'it': 'ita',
            'pt': 'por', 'ru': 'rus', 'ja': 'jpn', 'ko': 'kor', 'zh': 'chi',
            'ar': 'ara', 'hi': 'hin', 'nl': 'dut', 'sv': 'swe', 'no': 'nor',
            'da': 'dan', 'fi': 'fin', 'pl': 'pol', 'cs': 'cze', 'hu': 'hun',
            'el': 'gre', 'tr': 'tur', 'he': 'heb', 'th': 'tha', 'vi': 'vie',
            'uk': 'ukr', 'bg': 'bul', 'ro': 'rum', 'sk': 'slo', 'sl': 'slv',
            'sr': 'srp', 'hr': 'hrv', 'bs': 'bos', 'sq': 'alb', 'mk': 'mac',
            'lt': 'lit', 'lv': 'lav', 'et': 'est', 'mt': 'mlt', 'is': 'ice',
            'ga': 'gle', 'cy': 'wel', 'eu': 'baq', 'ca': 'cat', 'gl': 'glg',
            'fa': 'per', 'ur': 'urd', 'bn': 'ben', 'gu': 'guj', 'pa': 'pan',
            'ta': 'tam', 'te': 'tel', 'kn': 'kan', 'ml': 'mal', 'mr': 'mar',
            'ne': 'nep', 'si': 'sin', 'my': 'bur', 'km': 'khm', 'lo': 'lao',
            'bo': 'tib', 'mn': 'mon', 'kk': 'kaz', 'uz': 'uzb', 'ky': 'kir',
            'tg': 'tgk', 'tk': 'tuk', 'az': 'aze', 'hy': 'arm', 'ka': 'geo',
            'am': 'amh', 'sw': 'swa', 'yo': 'yor', 'ig': 'ibo', 'ha': 'hau',
            'so': 'som', 'af': 'afr', 'zu': 'zul', 'xh': 'xho', 'ms': 'may',
            'id': 'ind', 'tl': 'tgl', 'jv': 'jav', 'su': 'sun', 'eo': 'epo',
            'la': 'lat'
        }
        
        return iso639_1_to_2_mapping.get(code.lower(), code)

    def _detect_language_by_characters(self, text: str, subtitle_count: int) -> Optional[Dict]:
        """Fallback language detection using character analysis with improved confidence calculation."""
        
        if not text or len(text.strip()) < 10:
            # Not enough text for reliable detection
            return {
                'language_code': 'und',
                'confidence': 0.0,
                'subtitle_count': subtitle_count
            }
        
        # Count character types
        latin_chars = sum(1 for c in text if ord(c) < 0x0250)
        cyrillic_chars = sum(1 for c in text if 0x0400 <= ord(c) <= 0x04FF)
        arabic_chars = sum(1 for c in text if 0x0600 <= ord(c) <= 0x06FF)
        cjk_chars = sum(1 for c in text if 0x4E00 <= ord(c) <= 0x9FFF)
        
        total_chars = len(text.replace(' ', '').replace('\n', ''))  # Count only non-whitespace
        
        if total_chars == 0:
            return {
                'language_code': 'und',
                'confidence': 0.0,
                'subtitle_count': subtitle_count
            }
        
        # Calculate confidence based on character distribution strength
        cyrillic_ratio = cyrillic_chars / total_chars
        arabic_ratio = arabic_chars / total_chars
        cjk_ratio = cjk_chars / total_chars
        latin_ratio = latin_chars / total_chars
        
        # Determine language based on character distribution with dynamic confidence
        if cyrillic_ratio > 0.3:
            # Confidence increases with higher ratio
            confidence = min(0.9, 0.5 + cyrillic_ratio * 0.5)
            return {'language_code': 'rus', 'confidence': confidence, 'subtitle_count': subtitle_count}
        elif arabic_ratio > 0.3:
            confidence = min(0.9, 0.5 + arabic_ratio * 0.5)
            return {'language_code': 'ara', 'confidence': confidence, 'subtitle_count': subtitle_count}
        elif cjk_ratio > 0.3:
            confidence = min(0.85, 0.45 + cjk_ratio * 0.5)
            return {'language_code': 'chi', 'confidence': confidence, 'subtitle_count': subtitle_count}
        elif latin_ratio > 0.7:
            # Lower confidence for Latin script as it could be many languages
            # Confidence based on text length and ratio
            base_confidence = 0.3
            ratio_bonus = (latin_ratio - 0.7) * 0.3  # Up to 0.09 bonus
            length_bonus = min(0.2, len(text) / 5000)  # Up to 0.2 bonus for longer text
            confidence = min(0.65, base_confidence + ratio_bonus + length_bonus)
            return {'language_code': 'eng', 'confidence': confidence, 'subtitle_count': subtitle_count}
        
        # If no clear script detected, return undefined with very low confidence
        return {
            'language_code': 'und',
            'confidence': 0.1,
            'subtitle_count': subtitle_count
        }

    def detect_subtitle_language(self, subtitle_path: Path, file_path: Path = None, 
                                subtitle_track_index: int = None, stream_index: int = None) -> Optional[Dict]:
        """Detect language of subtitle file using text analysis or OCR."""
        try:
            # Check file extension to determine subtitle type
            if subtitle_path.suffix.lower() == '.sup':
                # Image-based subtitle (PGS format)
                if self.config.show_details:
                    logger.warning("Image-based subtitle detected (PGS/SUP)")
                    logger.warning("Language detection for image-based subtitles requires OCR")
                    logger.warning("Install pytesseract and tesseract-ocr for image subtitle support:")
                    logger.warning("  pip install pytesseract pillow")
                    logger.warning("  https://github.com/tesseract-ocr/tesseract")
                
                # Try OCR if available
                try:
                    if file_path and subtitle_track_index is not None and stream_index is not None:
                        return self._extract_pgs_images_and_ocr(file_path, subtitle_track_index, stream_index)
                    else:
                        return self._detect_language_via_ocr(subtitle_path)
                except ImportError:
                    if self.config.show_details:
                        logger.error("OCR libraries not available - skipping image-based subtitle")
                    return None
            else:
                # Text-based subtitle (SRT format)
                subtitles = self.parse_srt_file(subtitle_path)
                
                if not subtitles:
                    if self.config.show_details:
                        logger.warning("No subtitle entries found")
                    return None
                
                # Get text sample
                text_sample = self.get_subtitle_text_sample(subtitles)
                
                if not text_sample or len(text_sample.strip()) < 50:
                    if self.config.show_details:
                        logger.warning("Insufficient subtitle text for language detection")
                    # Return low confidence result for empty/minimal content
                    return {
                        'language_code': 'und',
                        'confidence': 0.0,
                        'subtitle_count': len(subtitles)
                    }
                
                if self.config.show_details:
                    logger.info(f"Analyzing {len(subtitles)} subtitle entries ({len(text_sample)} characters)")
                
                # Try to use langdetect library
                try:
                    import langdetect
                    from langdetect import detect_langs
                    
                    # Set seed for consistent results
                    langdetect.DetectorFactory.seed = 0
                    
                    # Detect language with confidence scores
                    detected_langs = detect_langs(text_sample)
                    
                    if detected_langs:
                        primary_lang = detected_langs[0]
                        language_code = primary_lang.lang
                        confidence = primary_lang.prob
                        
                        # Convert ISO 639-1 to ISO 639-2
                        language_code = self._convert_iso639_1_to_2(language_code)
                        
                        if self.config.show_details:
                            logger.info(f"Detected subtitle language: {language_code} (confidence: {confidence:.2f})")
                            if len(detected_langs) > 1:
                                logger.info(f"Other possibilities: {[(l.lang, f'{l.prob:.2f}') for l in detected_langs[1:3]]}")
                        
                        return {
                            'language_code': language_code,
                            'confidence': confidence,
                            'subtitle_count': len(subtitles)
                        }
                    else:
                        if self.config.show_details:
                            logger.warning("langdetect returned no results")
                        # Fallback to character analysis
                        return self._detect_language_by_characters(text_sample, len(subtitles))
                    
                except ImportError:
                    if self.config.show_details:
                        logger.warning("langdetect library not installed. Install with: pip install langdetect")
                        logger.warning("Falling back to basic text analysis")
                    
                    # Fallback: Use basic character analysis
                    return self._detect_language_by_characters(text_sample, len(subtitles))
                
                except Exception as e:
                    if self.config.show_details:
                        logger.warning(f"Language detection failed: {e}")
                    # Fallback to character analysis
                    return self._detect_language_by_characters(text_sample, len(subtitles))
                    
        except Exception as e:
            logger.error(f"Error detecting subtitle language: {e}")
            return None
    
    def _detect_language_via_ocr(self, subtitle_path: Path) -> Optional[Dict]:
        """Detect language from image-based subtitles using OCR."""
        try:
            import pytesseract
            from PIL import Image
            
            logger.info("Attempting OCR on image-based subtitles...")
            
            # For PGS subtitles embedded in MKV, we need to extract directly from the video file
            # Not from the already-extracted .sup file
            return None  # Signal that we need a different approach
            
        except ImportError:
            logger.error("pytesseract or PIL not installed")
            logger.error("Install with: pip install pytesseract pillow")
            return None
        except Exception as e:
            logger.error(f"OCR detection failed: {e}")
            return None

    def _extract_pgs_images_and_ocr(self, file_path: Path, subtitle_track_index: int, stream_index: int) -> Optional[Dict]:
        """Extract PGS subtitle images directly from video file and perform OCR."""
        try:
            import pytesseract
            from PIL import Image
    
            if sys.platform == 'win32':
                # Common Tesseract installation paths on Windows
                tesseract_paths = [
                    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                    r"C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe".format(os.getenv('USERNAME')),
                ]
                
                for path in tesseract_paths:
                    if os.path.exists(path):
                        pytesseract.pytesseract.tesseract_cmd = path
                        break
    
            # Check if Tesseract is available
            try:
                pytesseract.get_tesseract_version()
            except:
                logger.error("Tesseract OCR not found. Please install:")
                logger.error("Windows: https://github.com/UB-Mannheim/tesseract/wiki")
                logger.error("Linux: apt-get install tesseract-ocr")
                logger.error("Mac: brew install tesseract")
                return None
            
            logger.info("Extracting subtitle images for OCR analysis...")
            
            temp_dir = tempfile.mkdtemp()
            try:
                # Method 1: Try extracting using subtitle filter
                cmd = [
                    self.ffmpeg, '-y', '-v', 'warning',
                    '-i', str(file_path),
                    '-filter_complex', f'[0:s:{subtitle_track_index}]scale=iw:ih[sub]',
                    '-map', '[sub]',
                    '-frames:v', '50',  # Extract up to 50 subtitle frames
                    '-vsync', '0',  # Don't duplicate frames
                    f'{temp_dir}/sub_%04d.png'
                ]
                
                limited_cmd = limit_subprocess_resources(cmd)
                
                if self.config.show_details:
                    logger.info(f"Attempting Method 1: subtitle filter extraction")
                
                try:
                    result = subprocess.run(limited_cmd, capture_output=True, text=True, 
                                          encoding='utf-8', errors='replace', timeout=120)
                except subprocess.TimeoutExpired:
                    logger.warning("Method 1 timed out")
                
                # Check if any images were extracted
                image_files = list(Path(temp_dir).glob('sub_*.png'))
                
                # Method 2: If Method 1 failed, try direct stream copy and conversion
                if not image_files:
                    if self.config.show_details:
                        logger.info("Method 1 failed, trying Method 2: direct stream extraction")
                    
                    # First extract the subtitle stream to a temporary file
                    temp_sub_file = Path(temp_dir) / "subtitles.sup"
                    
                    cmd = [
                        self.ffmpeg, '-y', '-v', 'warning',
                        '-i', str(file_path),
                        '-map', f'0:s:{subtitle_track_index}',
                        '-c', 'copy',
                        str(temp_sub_file)
                    ]
                    
                    limited_cmd = limit_subprocess_resources(cmd)
                    
                    try:
                        result = subprocess.run(limited_cmd, capture_output=True, text=True,
                                              encoding='utf-8', errors='replace', timeout=60)
                        
                        if temp_sub_file.exists() and temp_sub_file.stat().st_size > 0:
                            # Now try to extract images from the SUP file
                            cmd = [
                                self.ffmpeg, '-y', '-v', 'warning',
                                '-i', str(temp_sub_file),
                                '-frames:v', '50',
                                '-vsync', '0',
                                f'{temp_dir}/sub_%04d.png'
                            ]
                            
                            limited_cmd = limit_subprocess_resources(cmd)
                            
                            try:
                                result = subprocess.run(limited_cmd, capture_output=True, text=True,
                                                      encoding='utf-8', errors='replace', timeout=120)
                            except subprocess.TimeoutExpired:
                                logger.warning("Method 2 timed out")
                            
                            image_files = list(Path(temp_dir).glob('sub_*.png'))
                            
                    except subprocess.CalledProcessError as e:
                        if self.config.show_details:
                            logger.debug(f"Method 2 failed: {e}")
                
                # Method 3: Use BDSup2Sub if available (external tool for PGS extraction)
                if not image_files:
                    if self.config.show_details:
                        logger.info("Method 2 failed, trying Method 3: frame extraction with overlay")
                    
                    # Extract video frames where subtitles appear
                    cmd = [
                        self.ffmpeg, '-y', '-v', 'warning',
                        '-i', str(file_path),
                        '-filter_complex', f'[0:v][0:s:{subtitle_track_index}]overlay[v]',
                        '-map', '[v]',
                        '-frames:v', '50',
                        '-vsync', '0',
                        '-q:v', '2',  # High quality
                        f'{temp_dir}/sub_%04d.png'
                    ]
                    
                    limited_cmd = limit_subprocess_resources(cmd)
                    
                    try:
                        result = subprocess.run(limited_cmd, capture_output=True, text=True,
                                              encoding='utf-8', errors='replace', timeout=120)
                    except subprocess.TimeoutExpired:
                        logger.warning("Method 3 timed out")
                    
                    image_files = list(Path(temp_dir).glob('sub_*.png'))
                
                if not image_files:
                    logger.warning("No subtitle images extracted using any method")
                    logger.warning("PGS subtitle extraction is complex and may not work with all files")
                    logger.warning("Consider using external tools like BDSup2Sub for PGS subtitle extraction")
                    return None
                
                if self.config.show_details:
                    logger.info(f"Extracted {len(image_files)} subtitle images")
                
                # OCR the extracted images
                ocr_texts = []
                successful_ocr = 0
                
                for img_file in image_files[:30]:  # Process up to 30 images
                    try:
                        img = Image.open(img_file)
                        
                        # Preprocess image for better OCR
                        # Convert to grayscale and increase contrast
                        img = img.convert('L')
                        
                        # Enhance contrast
                        from PIL import ImageEnhance
                        enhancer = ImageEnhance.Contrast(img)
                        img = enhancer.enhance(2.0)
                        
                        # OCR with multiple language support
                        text = pytesseract.image_to_string(img, config='--psm 6')
                        
                        if text.strip() and len(text.strip()) > 2:
                            ocr_texts.append(text.strip())
                            successful_ocr += 1
                            
                            if self.config.show_details and successful_ocr <= 3:
                                logger.info(f"OCR sample {successful_ocr}: '{text.strip()[:50]}...'")
                                
                    except Exception as e:
                        if self.config.show_details:
                            logger.debug(f"Failed to OCR {img_file.name}: {e}")
                        continue
                
                if not ocr_texts:
                    logger.warning("No text extracted from subtitle images via OCR")
                    logger.warning("This could mean:")
                    logger.warning("  1. The subtitles are in a language not supported by Tesseract")
                    logger.warning("  2. The image quality is too low for OCR")
                    logger.warning("  3. The extraction method didn't capture subtitle content properly")
                    return None
                
                logger.info(f"Successfully OCR'd {successful_ocr} subtitle images")
                
                # Combine OCR texts
                combined_text = ' '.join(ocr_texts)
                
                if len(combined_text.strip()) < 50:
                    logger.warning(f"Insufficient OCR text for detection ({len(combined_text)} chars)")
                    return None
                
                # Detect language using langdetect
                try:
                    import langdetect
                    from langdetect import detect_langs
                    
                    detected_langs = detect_langs(combined_text)
                    
                    if detected_langs:
                        primary_lang = detected_langs[0]
                        language_code = self._convert_iso639_1_to_2(primary_lang.lang)
                        
                        # Lower confidence for OCR-based detection
                        confidence = primary_lang.prob * 0.75  # Reduce confidence due to OCR uncertainty
                        
                        logger.info(f"OCR detected language: {language_code} (confidence: {confidence:.2f})")
                        
                        return {
                            'language_code': language_code,
                            'confidence': confidence,
                            'subtitle_count': len(ocr_texts)
                        }
                    else:
                        logger.warning("Language detection returned no results")
                        return None
                        
                except ImportError:
                    logger.warning("langdetect not available for OCR text analysis")
                    return self._detect_language_by_characters(combined_text, len(ocr_texts))
                except Exception as e:
                    logger.warning(f"Language detection failed: {e}")
                    return self._detect_language_by_characters(combined_text, len(ocr_texts))
                    
            finally:
                # Cleanup temp directory
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            
            return None
            
        except ImportError as e:
            logger.error(f"Required library not installed: {e}")
            logger.error("Install with: pip install pytesseract pillow")
            return None
        except Exception as e:
            logger.error(f"OCR detection failed: {e}")
            import traceback
            if self.config.show_details:
                traceback.print_exc()
            return None

    def _parse_srt_time(self, time_str: str) -> float:
        """Parse SRT timestamp to seconds."""
        # Format: HH:MM:SS,mmm
        time_str = time_str.replace(',', '.')
        parts = time_str.split(':')
        
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        
        return hours * 3600 + minutes * 60 + seconds

    def detect_forced_subtitles(self, file_path: Path, subtitle_track_index: int, stream_index: int, 
                               audio_track_index: int = 0, subtitle_path: Path = None) -> bool:
        """
        Detect if subtitle track is forced (only shows for foreign language parts).
        Uses multi-factor heuristic analysis to minimize expensive audio processing.
        """
        try:
            # Extract or use provided subtitle track
            if subtitle_path is None:
                subtitle_path = self.extract_subtitle_track(file_path, subtitle_track_index, stream_index)
                should_cleanup = True
            else:
                should_cleanup = False
                
            if not subtitle_path:
                if self.config.show_details:
                    logger.warning("Could not extract subtitle track for forced detection")
                return False
            
            try:
                # Check if it's an image-based subtitle (PGS/SUP)
                is_image_based = subtitle_path.suffix.lower() == '.sup'
                
                if is_image_based:
                    # For image-based subtitles, use frame count heuristics
                    return self._detect_forced_pgs_subtitles(file_path, subtitle_track_index, 
                                                            stream_index, audio_track_index)
                
                # For text-based subtitles, use multi-factor analysis
                subtitles = self.parse_srt_file(subtitle_path)
                
                if not subtitles:
                    if self.config.show_details:
                        logger.warning("No subtitle entries found")
                    return False
                
                # Get file duration
                duration = self.get_file_duration(file_path)
                if duration <= 0:
                    if self.config.show_details:
                        logger.warning("Could not determine file duration")
                    return False
                
                duration_minutes = duration / 60
                
                # Calculate comprehensive statistics
                stats = self._calculate_subtitle_statistics(subtitles, duration)
                
                if self.config.show_details:
                    logger.info(f"Subtitle statistics:")
                    logger.info(f"  Count: {stats['count']} subtitles")
                    logger.info(f"  Density: {stats['density']:.1f} subs/min")
                    logger.info(f"  Coverage: {stats['coverage_percent']:.1f}% of duration")
                    logger.info(f"  Avg duration: {stats['avg_duration']:.1f}s per subtitle")
                
                # Make decision using multi-factor analysis
                decision, reason, confidence = self._decide_forced_from_statistics(stats, duration_minutes)
                
                # High or medium confidence - use the decision
                if confidence >= 2:
                    if self.config.show_details:
                        confidence_label = "HIGH" if confidence == 3 else "MEDIUM"
                        logger.info(f"{confidence_label} confidence decision: {'FORCED' if decision else 'FULL'}")
                        logger.info(f"  Reason: {reason}")
                    return decision
                
                # Low confidence - need audio analysis
                if self.config.show_details:
                    logger.info(f"LOW confidence: {reason}")
                    logger.info("Analyzing audio for speech patterns...")
                
                # Extract FULL audio track for accurate analysis (with timeout protection)
                try:
                    audio_path = self.extract_full_audio_track(file_path, audio_track_index, stream_index)
                    if not audio_path:
                        if self.config.show_details:
                            logger.warning("Could not extract audio, using heuristic fallback")
                        # Fallback heuristic
                        is_forced = stats['density'] < 5.5 or stats['coverage_percent'] < 37.5
                        return is_forced
                except Exception as e:
                    logger.error(f"Audio extraction failed: {e}")
                    logger.warning("Using heuristic fallback")
                    is_forced = stats['density'] < 5.5 or stats['coverage_percent'] < 37.5
                    return is_forced
                
                try:
                    # Use Whisper to detect speech segments with VAD
                    if self.config.show_details:
                        logger.info("Running speech detection on full audio track...")
                    
                    try:
                        segments, info = self.whisper_model.transcribe(
                            str(audio_path),
                            language=None,
                            task="transcribe",
                            beam_size=1,
                            best_of=1,
                            temperature=0.0,
                            vad_filter=True,
                            vad_parameters={
                                "min_speech_duration_ms": 250,
                                "max_speech_duration_s": 30
                            },
                            word_timestamps=True
                        )
                    except Exception as e:
                        logger.error(f"Speech detection failed: {e}")
                        if audio_path and audio_path.exists():
                            audio_path.unlink()
                        # Fallback to heuristics
                        is_forced = stats['density'] < 5.5 or stats['coverage_percent'] < 37.5
                        return is_forced
                    
                    if self.config.show_details:
                        logger.info(f"Transcription completed, processing results...")
                    
                    # Collect speech timings
                    speech_timings = []
                    for segment in segments:
                        speech_timings.append((segment.start, segment.end))
                    
                    if not speech_timings:
                        if self.config.show_details:
                            logger.info("No speech detected in audio - likely forced or silent")
                        return True
                    
                    # Calculate speech coverage
                    total_speech_duration = sum(end - start for start, end in speech_timings)
                    speech_coverage = (total_speech_duration / duration) * 100
                    
                    # Calculate overlap between subtitles and speech
                    overlap_duration = 0.0
                    subtitle_timings = stats['subtitle_timings']
                    
                    for sub_start, sub_end in subtitle_timings:
                        for speech_start, speech_end in speech_timings:
                            overlap_start = max(sub_start, speech_start)
                            overlap_end = min(sub_end, speech_end)
                            if overlap_start < overlap_end:
                                overlap_duration += (overlap_end - overlap_start)
                    
                    speech_with_subtitles = (overlap_duration / total_speech_duration * 100) if total_speech_duration > 0 else 0
                    coverage_ratio = stats['coverage_percent'] / speech_coverage if speech_coverage > 0 else 0
                    
                    if self.config.show_details:
                        logger.info(f"Audio analysis results:")
                        logger.info(f"  Speech coverage: {speech_coverage:.1f}% of total duration")
                        logger.info(f"  Speech with subtitles: {speech_with_subtitles:.1f}%")
                        logger.info(f"  Coverage ratio (sub/speech): {coverage_ratio:.2f}")
                    
                    # Determine if forced based on audio analysis
                    is_forced = False
                    
                    if speech_with_subtitles < 50:
                        is_forced = True
                        if self.config.show_details:
                            logger.info(f"Only {speech_with_subtitles:.1f}% of speech has subtitles - likely forced")
                    
                    if coverage_ratio < 0.4:
                        is_forced = True
                        if self.config.show_details:
                            logger.info(f"Subtitle coverage much less than speech ({coverage_ratio:.2f}) - likely forced")
                    
                    if stats['coverage_percent'] < 25 and stats['density'] < 5:
                        is_forced = True
                        if self.config.show_details:
                            logger.info("Low coverage and density - likely forced")
                    
                    if speech_with_subtitles > 80:
                        is_forced = False
                        if self.config.show_details:
                            logger.info(f"Most speech has subtitles ({speech_with_subtitles:.1f}%) - not forced")
                    
                    return is_forced
                    
                finally:
                    if audio_path and audio_path.exists():
                        audio_path.unlink()
                
            finally:
                # Only clean up if we extracted it ourselves
                if should_cleanup and subtitle_path and subtitle_path.exists():
                    subtitle_path.unlink()
            
        except Exception as e:
            logger.error(f"Error detecting forced subtitles: {e}")
            if self.config.show_details:
                import traceback
                traceback.print_exc()
            return False

    def _detect_forced_pgs_subtitles(self, file_path: Path, subtitle_track_index: int, 
                                    stream_index: int, audio_track_index: int = 0) -> bool:
        """
        Detect if PGS/image-based subtitle track is forced using frame count analysis.
        
        PGS subtitles are stored as images, so we can't parse text timing.
        Instead, we analyze:
        1. Total number of subtitle frames
        2. Distribution of frames across the video duration
        3. Comparison with audio speech patterns (optional)
        """
        try:
            if self.config.show_details:
                logger.info("Analyzing PGS subtitle using frame count method...")
            
            # Get file duration
            duration = self.get_file_duration(file_path)
            if duration <= 0:
                if self.config.show_details:
                    logger.warning("Could not determine file duration")
                return False
            
            # Use ffprobe to count subtitle frames
            try:
                cmd = [
                    self.ffprobe, '-v', 'error',
                    '-select_streams', f's:{subtitle_track_index}',
                    '-count_packets',
                    '-show_entries', 'stream=nb_read_packets',
                    '-of', 'csv=p=0',
                    str(file_path)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, check=True, 
                                      encoding='utf-8', errors='replace')
                
                frame_count = int(result.stdout.strip())
                
                if self.config.show_details:
                    logger.info(f"PGS subtitle has {frame_count} frames")
                
            except (subprocess.CalledProcessError, ValueError) as e:
                if self.config.show_details:
                    logger.warning(f"Could not count PGS frames: {e}")
                return False
            
            # Calculate frames per minute
            frames_per_minute = frame_count / (duration / 60)
            
            if self.config.show_details:
                logger.info(f"PGS frame density: {frames_per_minute:.1f} frames/minute")
            
            # Heuristics for PGS forced subtitles:
            # - Forced subtitles typically have very few frames (< 100 for a movie)
            # - Or very low density (< 5 frames per minute)
            
            if frame_count < 100:
                if self.config.show_details:
                    logger.info(f"Very low frame count ({frame_count}) - likely forced subtitles")
                return True
            
            if frames_per_minute < 5:
                if self.config.show_details:
                    logger.info(f"Very low frame density ({frames_per_minute:.1f}/min) - likely forced subtitles")
                return True
            
            if frames_per_minute > 30:
                if self.config.show_details:
                    logger.info(f"High frame density ({frames_per_minute:.1f}/min) - not forced subtitles")
                return False
            
            # For borderline cases (5-30 frames/min), we could optionally analyze audio
            # but that's expensive, so use conservative threshold
            if frames_per_minute < 15:
                if self.config.show_details:
                    logger.info(f"Low-to-moderate frame density ({frames_per_minute:.1f}/min) - likely forced subtitles")
                return True
            
            if self.config.show_details:
                logger.info(f"Moderate frame density ({frames_per_minute:.1f}/min) - probably not forced")
            return False
            
        except Exception as e:
            logger.error(f"Error detecting forced PGS subtitles: {e}")
            if self.config.show_details:
                import traceback
                traceback.print_exc()
            return False

    def detect_sdh_subtitles(self, subtitle_path: Path) -> bool:
            """
            Detect if subtitles are SDH (Subtitles for the Deaf and Hard of Hearing).
            SDH subtitles contain sound effect descriptions and speaker identifications.
            """
            try:
                subtitles = self.parse_srt_file(subtitle_path)
                
                if not subtitles:
                    return False
                
                # Must contain letters/words, not just styling tags
                sdh_patterns = [
                    r'\[[A-Za-z\s]{3,}\]',  # [door closes], [music playing] - at least 3 letters
                    r'\([A-Za-z\s]{3,}\)',  # (sighs), (footsteps) - at least 3 letters
                    r'♪[^♪]+♪',              # ♪ music ♪
                    r'\*[A-Za-z\s]{3,}\*',  # *laughs* - at least 3 letters
                ]
                
                sdh_indicator_count = 0
                total_text_length = 0
                
                # Common SDH keywords that appear in brackets/parentheses
                sdh_keywords = [
                    'narrator', 'narrating', 'speaking', 'whispering', 'shouting', 'yelling', 'screaming',
                    'music', 'playing', 'door', 'closes', 'opens', 'phone', 'ringing', 'rings',
                    'footsteps', 'sighs', 'sigh', 'laughs', 'laugh', 'cries', 'cry', 'crying',
                    'knocking', 'knock', 'barking', 'bark', 'meowing', 'beeping', 'beep',
                    'gunshot', 'explosion', 'thunder', 'applause', 'cheering', 'clapping',
                    'breathing', 'coughing', 'snoring', 'groaning', 'grunting',
                    'chatter', 'chattering', 'murmuring', 'rustling', 'creaking',
                    'dramatic music', 'tense music', 'suspenseful music', 'upbeat music',
                    'in distance', 'muffled', 'echoing', 'faintly'
                ]
                
                for sub in subtitles:
                    text = sub['text']
                    total_text_length += len(text)
                    
                    # Check if any pattern matches
                    has_sdh_indicator = False
                    for pattern in sdh_patterns:
                        matches = re.findall(pattern, text, re.IGNORECASE)
                        if matches:
                            # Verify that the match contains actual SDH keywords
                            for match in matches:
                                # Extract just the text content (remove brackets/parentheses)
                                content = re.sub(r'[\[\]\(\)\*♪]', '', match).strip().lower()
                                
                                # Check if it contains SDH keywords
                                if any(keyword in content for keyword in sdh_keywords):
                                    has_sdh_indicator = True
                                    break
                        
                        if has_sdh_indicator:
                            break
                    
                    if has_sdh_indicator:
                        sdh_indicator_count += 1
                
                # Calculate SDH indicator ratio
                sdh_ratio = sdh_indicator_count / len(subtitles) if subtitles else 0
                
                if self.config.show_details:
                    logger.info(f"SDH indicators found in {sdh_indicator_count}/{len(subtitles)} subtitles ({sdh_ratio*100:.1f}%)")
                
                # If more than 10% of subtitles contain SDH indicators, classify as SDH
                # (Lowered from 15% since we're now more specific)
                is_sdh = sdh_ratio > 0.10
                
                # Additional check: Look for common SDH phrases in the full text
                # This catches cases where SDH content might not always be in brackets
                full_text = ' '.join([sub['text'].lower() for sub in subtitles])
                
                # Count phrases that appear in typical SDH contexts
                sdh_phrase_patterns = [
                    r'\bnarrator\b', r'\bspeaking\b', r'\bwhispering\b', r'\bshouting\b',
                    r'\bmusic playing\b', r'\bdoor closes\b', r'\bphone ringing\b',
                    r'\bfootsteps\b', r'\bsighs\b', r'\blaughs\b', r'\bcries\b',
                    r'\bin the distance\b', r'\bmuffled\b', r'\bechoing\b',
                    r'\bdramatic music\b', r'\btense music\b'
                ]
                
                sdh_phrase_count = sum(1 for pattern in sdh_phrase_patterns 
                                      if re.search(pattern, full_text))
                
                if sdh_phrase_count >= 3:
                    is_sdh = True
                    if self.config.show_details:
                        logger.info(f"Found {sdh_phrase_count} SDH phrase patterns")
                
                return is_sdh
                
            except Exception as e:
                logger.error(f"Error detecting SDH subtitles: {e}")
                return False

    def _get_language_name(self, language_code: str) -> str:
        """Convert language code to human-readable name."""
        language_names = {
            'eng': 'English', 'spa': 'Spanish', 'fre': 'French', 'ger': 'German',
            'ita': 'Italian', 'por': 'Portuguese', 'rus': 'Russian', 'jpn': 'Japanese',
            'chi': 'Chinese', 'kor': 'Korean', 'ara': 'Arabic', 'hin': 'Hindi',
            'dut': 'Dutch', 'swe': 'Swedish', 'nor': 'Norwegian', 'dan': 'Danish',
            'fin': 'Finnish', 'pol': 'Polish', 'cze': 'Czech', 'hun': 'Hungarian',
            'gre': 'Greek', 'tur': 'Turkish', 'heb': 'Hebrew', 'tha': 'Thai',
            'vie': 'Vietnamese', 'ukr': 'Ukrainian', 'bul': 'Bulgarian',
            'rum': 'Romanian', 'slo': 'Slovak', 'slv': 'Slovenian', 'srp': 'Serbian',
            'hrv': 'Croatian', 'bos': 'Bosnian', 'zxx': 'No Linguistic Content',
            'alb': 'Albanian', 'mac': 'Macedonian', 'lit': 'Lithuanian', 'lav': 'Latvian',
            'est': 'Estonian', 'mlt': 'Maltese', 'ice': 'Icelandic', 'gle': 'Irish',
            'wel': 'Welsh', 'baq': 'Basque', 'cat': 'Catalan', 'glg': 'Galician',
            'per': 'Persian', 'urd': 'Urdu', 'ben': 'Bengali', 'guj': 'Gujarati',
            'pan': 'Punjabi', 'tam': 'Tamil', 'tel': 'Telugu', 'kan': 'Kannada',
            'mal': 'Malayalam', 'mar': 'Marathi', 'nep': 'Nepali', 'sin': 'Sinhalese',
            'bur': 'Burmese', 'khm': 'Khmer', 'lao': 'Lao', 'tib': 'Tibetan',
            'mon': 'Mongolian', 'kaz': 'Kazakh', 'uzb': 'Uzbek', 'kir': 'Kyrgyz',
            'tgk': 'Tajik', 'tuk': 'Turkmen', 'aze': 'Azerbaijani', 'arm': 'Armenian',
            'geo': 'Georgian', 'amh': 'Amharic', 'swa': 'Swahili', 'yor': 'Yoruba',
            'ibo': 'Igbo', 'hau': 'Hausa', 'som': 'Somali', 'afr': 'Afrikaans',
            'zul': 'Zulu', 'xho': 'Xhosa', 'may': 'Malay', 'ind': 'Indonesian',
            'tgl': 'Tagalog', 'jav': 'Javanese', 'sun': 'Sundanese', 'epo': 'Esperanto',
            'lat': 'Latin'
        }
        
        return language_names.get(language_code, language_code.upper())

    def update_subtitle_metadata(self, file_path: Path, track_index: int, language_code: str,
                                is_forced: bool = False, is_sdh: bool = False,
                                dry_run: bool = False) -> bool:
        """Update subtitle track metadata including language, name, and flags."""
        
        # Build track name
        language_name = self._get_language_name(language_code)
        track_name_parts = [language_name]
        
        if is_forced:
            track_name_parts.append("[Forced]")
        if is_sdh:
            track_name_parts.append("[SDH]")
        
        track_name = " ".join(track_name_parts)
        
        if dry_run:
            forced_flag = " (forced)" if is_forced else ""
            sdh_flag = " (SDH)" if is_sdh else ""
            print(f"[DRY RUN] Would update subtitle track {track_index} in {file_path.name}:")
            print(f"           Language: {language_code}, Name: '{track_name}'{forced_flag}{sdh_flag}")
            return True
        
        try:
            # Build mkvpropedit command
            cmd = [
                self.mkvpropedit, str(file_path),
                '--edit', f'track:s{track_index + 1}',
                '--set', f'language={language_code}',
                '--set', f'name={track_name}'
            ]
            
            # Add forced flag if applicable
            if is_forced:
                cmd.extend(['--set', 'flag-forced=1'])
            else:
                cmd.extend(['--set', 'flag-forced=0'])
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            if self.config.show_details:
                logger.info(f"Updated subtitle track {track_index} in {file_path.name}")
                logger.info(f"  Language: {language_code}, Name: '{track_name}'")
                if is_forced:
                    logger.info(f"  Marked as forced")
                if is_sdh:
                    logger.info(f"  Marked as SDH")
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error updating subtitle metadata for {file_path}: {e}")
            return False

    def process_subtitle_tracks(self, file_path: Path) -> Dict:
        """Process all subtitle tracks in a file."""
        results = {
            'subtitle_tracks_found': 0,
            'processed_subtitle_tracks': [],
            'failed_subtitle_tracks': [],
            'skipped_subtitle_tracks': [],
            'subtitle_errors': []
        }
        
        if not self.config.process_subtitles:
            return results
        
        # Find subtitle tracks
        if self.config.reprocess_all_subtitles:
            subtitle_tracks = self.find_all_subtitle_tracks(file_path)
            track_type = "all subtitle"
        else:
            undefined_tracks = self.find_undefined_subtitle_tracks(file_path)
            subtitle_tracks = []
            for subtitle_track_index, stream_info, stream_index in undefined_tracks:
                subtitle_tracks.append((subtitle_track_index, stream_info, stream_index, 'und'))
            track_type = "undefined subtitle"
        
        results['subtitle_tracks_found'] = len(subtitle_tracks)
        
        if not subtitle_tracks:
            if self.config.show_details:
                logger.info(f"No {track_type} tracks found in {file_path.name}")
            return results
        
        if self.config.show_details:
            print(f"Found {len(subtitle_tracks)} {track_type} track(s) to process")
        
        # Process each subtitle track
        for track_data in subtitle_tracks:
            if len(track_data) == 4:
                subtitle_track_index, stream_info, stream_index, current_language = track_data
            else:
                subtitle_track_index, stream_info, stream_index = track_data
                current_language = 'und'
            
            # Extract subtitle ONCE and reuse for all analyses
            subtitle_path = None
            
            try:
                if self.config.show_details:
                    print(f"Processing subtitle track {subtitle_track_index}...")
                
                # Extract subtitle (ONCE)
                subtitle_path = self.extract_subtitle_track(file_path, subtitle_track_index, stream_index)
                if not subtitle_path:
                    results['failed_subtitle_tracks'].append(subtitle_track_index)
                    results['subtitle_errors'].append(f"Failed to extract subtitle track {subtitle_track_index}")
                    continue
                
                # Detect language - pass the already extracted subtitle path
                language_result = self.detect_subtitle_language(
                    subtitle_path, 
                    file_path=file_path,
                    subtitle_track_index=subtitle_track_index,
                    stream_index=stream_index
                )
                
                if not language_result:
                    results['failed_subtitle_tracks'].append(subtitle_track_index)
                    results['subtitle_errors'].append(f"Failed to detect language for subtitle track {subtitle_track_index}")
                    continue
                
                language_code = language_result['language_code']
                confidence = language_result['confidence']
                
                # Check confidence threshold - SKIP instead of prompting
                if confidence < self.config.subtitle_confidence_threshold:
                    logger.warning(f"Subtitle track {subtitle_track_index} confidence ({confidence:.2f}) below threshold ({self.config.subtitle_confidence_threshold}) - skipping")
                    results['skipped_subtitle_tracks'].append({
                        'track_index': subtitle_track_index,
                        'detected_language': language_code,
                        'confidence': confidence,
                        'reason': 'confidence_below_threshold'
                    })
                    continue
                
                # Detect forced subtitles (reuse extracted subtitle)
                is_forced = False
                if self.config.analyze_forced_subtitles:
                    if self.config.show_details:
                        print(f"Analyzing if subtitle track {subtitle_track_index} is forced...")
                    is_forced = self.detect_forced_subtitles(
                        file_path, subtitle_track_index, stream_index,
                        subtitle_path=subtitle_path  # Pass the already extracted subtitle
                    )
                
                # Detect SDH (reuse extracted subtitle)
                is_sdh = False
                if self.config.detect_sdh_subtitles:
                    if self.config.show_details:
                        print(f"Analyzing if subtitle track {subtitle_track_index} is SDH...")
                    is_sdh = self.detect_sdh_subtitles(subtitle_path)
                
                # Update metadata
                success = self.update_subtitle_metadata(
                    file_path, subtitle_track_index, language_code,
                    is_forced, is_sdh, self.config.dry_run
                )
                
                if success:
                    results['processed_subtitle_tracks'].append({
                        'track_index': subtitle_track_index,
                        'detected_language': language_code,
                        'previous_language': current_language,
                        'confidence': confidence,
                        'is_forced': is_forced,
                        'is_sdh': is_sdh
                    })
                else:
                    results['failed_subtitle_tracks'].append(subtitle_track_index)
                    results['subtitle_errors'].append(f"Failed to update subtitle track {subtitle_track_index}")
                
            except Exception as e:
                error_msg = f"Error processing subtitle track {subtitle_track_index}: {str(e)}"
                logger.error(error_msg)
                results['failed_subtitle_tracks'].append(subtitle_track_index)
                results['subtitle_errors'].append(error_msg)
            
            finally:
                # Clean up temporary subtitle file
                if subtitle_path and subtitle_path.exists():
                    subtitle_path.unlink()
        
        return results
			
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
            'errors': [],
            'subtitle_results': None,
            'skipped_due_to_tracking': False
        }
        
        # Check if file should be skipped due to tracking
        # If reprocess_all or reprocess_all_subtitles is enabled, tracking is bypassed
        should_skip_audio = False
        should_skip_subtitles = False
        
        if self.config.use_tracking and hasattr(self, 'tracker'):
            # Tracking is bypassed if reprocess_all or reprocess_all_subtitles is enabled
            bypass_tracking = self.config.reprocess_all or self.config.reprocess_all_subtitles
            
            if not bypass_tracking and self.tracker.is_processed(file_path):
                entry_key = str(file_path.absolute())
                entry = self.tracker.data.get(entry_key, {})
                
                # Check what was previously processed
                audio_was_processed = entry.get('audio_processed', False)
                subtitle_was_processed = entry.get('subtitle_processed', False)
                
                # Skip audio if it was processed (reprocess_all would have bypassed this check)
                if audio_was_processed:
                    should_skip_audio = True
                
                # Skip subtitles if they were processed (reprocess_all_subtitles would have bypassed this check)
                if subtitle_was_processed:
                    should_skip_subtitles = True
                
                # If both should be skipped, skip the entire file
                if should_skip_audio and (should_skip_subtitles or not self.config.process_subtitles):
                    if self.config.show_details:
                        logger.info(f"Skipping {file_path.name} (already processed)")
                    else:
                        print(f"Skipping {file_path.name} (already processed)")
                    results['skipped_due_to_tracking'] = True
                    return results
        
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
        
        # Track success for each processing type
        audio_success = False
        subtitle_success = False
        
        # Process audio tracks (unless skipped by tracking)
        if not should_skip_audio:
            # Find audio tracks based on config
            if self.config.reprocess_all:
                audio_tracks = self.find_all_audio_tracks(mkv_path)
                track_type = "all audio"
            else:
                undefined_tracks = self.find_undefined_audio_tracks(mkv_path)
                audio_tracks = []
                for audio_track_index, stream_info, stream_index in undefined_tracks:
                    audio_tracks.append((audio_track_index, stream_info, stream_index, 'und'))
                track_type = "undefined audio"
            
            results['undefined_tracks'] = len(audio_tracks)
            
            if not audio_tracks:
                if self.config.show_details:
                    logger.info(f"No {track_type} tracks found in {mkv_path.name}")
                # No undefined tracks = success
                audio_success = True
            else:
                # Process each track
                audio_had_failures = False
                for track_data in audio_tracks:
                    if len(track_data) == 4:
                        audio_track_index, stream_info, stream_index, current_language = track_data
                    else:
                        audio_track_index, stream_info, stream_index = track_data
                        current_language = 'und'
                        
                    try:
                        language_code = self.detect_language_with_retries(mkv_path, audio_track_index, stream_index, max_retries=3)
                        if not language_code:
                            results['failed_tracks'].append(audio_track_index)
                            results['errors'].append(f"Failed to detect language for track {audio_track_index}")
                            audio_had_failures = True
                            continue
                        
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
                            audio_had_failures = True
                            
                    except Exception as e:
                        error_msg = f"Error processing track {audio_track_index}: {str(e)}"
                        logger.error(error_msg)
                        results['failed_tracks'].append(audio_track_index)
                        results['errors'].append(error_msg)
                        audio_had_failures = True
                
                # Audio processing successful if we had tracks and no failures
                audio_success = not audio_had_failures
        else:
            # Skipped audio due to tracking
            audio_success = True  # Consider it successful since it was previously processed
        
        # Process subtitle tracks (unless skipped by tracking)
        if self.config.process_subtitles and not should_skip_subtitles:
            print(f"\nProcessing subtitles for: {mkv_path.name}")
            subtitle_results = self.process_subtitle_tracks(mkv_path)
            results['subtitle_results'] = subtitle_results
            
            # Subtitle processing successful if no failures
            subtitle_success = (
                len(subtitle_results.get('failed_subtitle_tracks', [])) == 0 and
                len(subtitle_results.get('subtitle_errors', [])) == 0
            )
        elif should_skip_subtitles:
            # Skipped subtitles due to tracking
            subtitle_success = True  # Consider it successful since it was previously processed
        
        # Mark as processed if successful and not dry run
        if (self.config.use_tracking and 
            hasattr(self, 'tracker') and 
            not self.config.dry_run):
            self.tracker.mark_processed(mkv_path, audio_success, subtitle_success)
        
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
        
        api_url = "https://api.github.com/repos/netplexflix/ULDAS/releases/latest"
        
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
                print("Download from: https://github.com/netplexflix/ULDAS")
                print(f"{'='*60}\n")
            else:
                print(f"✓ Up to date. Version: {VERSION}")
                
        except Exception as e:
            if latest_version != current_version:
                print("Update may be available")
                print(f"Current: {current_version}, Latest: {latest_version}")
                print("Check: https://github.com/netplexflix/ULDAS\n")
            else:
                print(f"✓ Up to date. Version: {VERSION}")
        
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

def print_detailed_summary(results: List[Dict], config: Config, runtime_seconds: float, 
                          detector: MKVLanguageDetector = None):
    print(f"\n{'='*60}")
    print("PROCESSING SUMMARY")
    print(f"{'='*60}")
    
    total_files = len(results)
    files_with_actions = 0
    files_successfully_processed = 0
    files_with_failures = 0
    files_processed = 0
    files_skipped = 0
    silent_content_files = []
    
    # ANSI color codes
    YELLOW = '\033[93m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    
    for result in results:
        if result.get('skipped_due_to_tracking'):
            files_skipped += 1
            continue
            
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
            
            if prev_lang == lang_code:
                if lang_code == 'zxx':
                    track_actions.append(f"{YELLOW}track{track_idx}: {prev_lang} -> {lang_code} (no speech){RESET}")
                    has_silent_content = True
                else:
                    track_actions.append(f"track{track_idx}: {prev_lang} -> {lang_code}")
            else:
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
    
    # Show summary
    if files_skipped > 0:
        print(f"\n{GREEN}Skipped {files_skipped} already-processed file(s){RESET}")
    
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
    
    # Show tracking statistics
    if config.use_tracking and detector and hasattr(detector, 'tracker'):
        stats = detector.tracker.get_stats()
        if stats['total_tracked'] > 0:
            print(f"\n{CYAN}Tracking Statistics:{RESET}")
            print(f"  Total files tracked: {stats['total_tracked']}")
            if stats['audio_only'] > 0:
                print(f"  Audio-only processed: {stats['audio_only']}")
            if stats['subtitle_only'] > 0:
                print(f"  Subtitle-only processed: {stats['subtitle_only']}")
            if stats['both'] > 0:
                print(f"  Both audio & subtitles: {stats['both']}")
    
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
    
    # Add subtitle summary
    if config.process_subtitles:
        print(f"\n{'='*60}")
        print("SUBTITLE PROCESSING SUMMARY")
        print(f"{'='*60}")
        
        total_subtitle_tracks = 0
        processed_subtitle_tracks = 0
        failed_subtitle_tracks = 0
        skipped_subtitle_tracks = 0
        forced_subtitles_found = 0
        sdh_subtitles_found = 0
        
        for result in results:
            if result.get('subtitle_results'):
                sub_results = result['subtitle_results']
                total_subtitle_tracks += sub_results['subtitle_tracks_found']
                processed_subtitle_tracks += len(sub_results['processed_subtitle_tracks'])
                failed_subtitle_tracks += len(sub_results['failed_subtitle_tracks'])
                skipped_subtitle_tracks += len(sub_results.get('skipped_subtitle_tracks', []))
                
                for track in sub_results['processed_subtitle_tracks']:
                    if track.get('is_forced'):
                        forced_subtitles_found += 1
                    if track.get('is_sdh'):
                        sdh_subtitles_found += 1
                
                # Print per-file subtitle details
                if (sub_results['processed_subtitle_tracks'] or 
                    sub_results['failed_subtitle_tracks'] or 
                    sub_results.get('skipped_subtitle_tracks')):
                    
                    file_name = Path(result['original_file']).name
                    print(f"\n{file_name}:")
                    
                    # Show processed tracks
                    for track in sub_results['processed_subtitle_tracks']:
                        track_idx = track['track_index']
                        lang = track['detected_language']
                        prev_lang = track['previous_language']
                        conf = track['confidence']
                        
                        flags = []
                        if track.get('is_forced'):
                            flags.append(f"{YELLOW}Forced{RESET}")
                        if track.get('is_sdh'):
                            flags.append(f"{CYAN}SDH{RESET}")
                        
                        flag_str = f" [{', '.join(flags)}]" if flags else ""
                        
                        if prev_lang == lang:
                            print(f"  subtitle track{track_idx}: {prev_lang} -> {lang} (conf: {conf:.2f}){flag_str}")
                        else:
                            print(f"  {CYAN}subtitle track{track_idx}: {prev_lang} -> {lang}{RESET} (conf: {conf:.2f}){flag_str}")
                    
                    # Show skipped tracks
                    if sub_results.get('skipped_subtitle_tracks'):
                        for track in sub_results['skipped_subtitle_tracks']:
                            track_idx = track['track_index']
                            lang = track['detected_language']
                            conf = track['confidence']
                            reason = track.get('reason', 'unknown')
                            
                            if reason == 'confidence_below_threshold':
                                reason_text = f"confidence below threshold ({config.subtitle_confidence_threshold})"
                            else:
                                reason_text = reason
                            
                            print(f"  {YELLOW}subtitle track{track_idx}: skipped{RESET} (detected: {lang}, conf: {conf:.2f}, reason: {reason_text})")
                    
                    # Show failed tracks
                    for track_idx in sub_results['failed_subtitle_tracks']:
                        print(f"  {RED}subtitle track{track_idx}: failed{RESET}")
        
        # Overall subtitle statistics
        if total_subtitle_tracks > 0:
            print(f"\nSubtitle tracks found: {total_subtitle_tracks}")
            print(f"Successfully processed: {processed_subtitle_tracks}")
            
            if skipped_subtitle_tracks > 0:
                print(f"{YELLOW}Skipped (low confidence): {skipped_subtitle_tracks}{RESET}")
            
            if failed_subtitle_tracks > 0:
                print(f"{RED}Failed: {failed_subtitle_tracks}{RESET}")
            
            if forced_subtitles_found > 0:
                print(f"Forced subtitles detected: {forced_subtitles_found}")
            
            if sdh_subtitles_found > 0:
                print(f"SDH subtitles detected: {sdh_subtitles_found}")
        else:
            print("\nNo subtitle tracks found to process")
    
    print(f"\nTotal runtime: {format_duration(runtime_seconds)}")
    
    if config.dry_run:
        print("(Dry run - no files were actually modified)")
    print()

def main():
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description='Detect and update language metadata for video file audio and subtitle tracks')
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
    parser.add_argument('--process-subtitles', action='store_true',
                       help='Process subtitle tracks in addition to audio tracks')
    parser.add_argument('--analyze-forced', action='store_true',
                       help='Analyze whether subtitle tracks are forced subtitles')
    parser.add_argument('--no-sdh-detection', action='store_true',
                       help='Disable SDH subtitle detection')
    parser.add_argument('--reprocess-all-subtitles', action='store_true',
                       help='Reprocess all subtitle tracks instead of only undefined ones')
    parser.add_argument('--force-reprocess', action='store_true',
                       help='Force reprocess all files, ignoring tracking cache')
    parser.add_argument('--clear-tracking', action='store_true',
                       help='Clear all tracking data and exit')
    parser.add_argument('--no-tracking', action='store_true',
                       help='Disable tracking feature for this run')
    
    args = parser.parse_args()
    
    # Handle tracking clear
    if args.clear_tracking:
        tracker = ProcessingTracker("config")
        stats = tracker.get_stats()
        tracker.clear_all()
        print(f"Cleared tracking data for {stats['total_tracked']} files")
        return
    
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
        config.path = [args.directory]
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
    if args.process_subtitles:
        config.process_subtitles = True
    if args.analyze_forced:
        config.analyze_forced_subtitles = True
    if args.no_sdh_detection:
        config.detect_sdh_subtitles = False
    if args.reprocess_all_subtitles:
        config.reprocess_all_subtitles = True
    if args.force_reprocess:
        config.force_reprocess = True
    if args.no_tracking:
        config.use_tracking = False
    
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
    
    # Check for optional langdetect library (for subtitle processing)
    if config.process_subtitles:
        try:
            import langdetect
        except ImportError:
            logger.warning("langdetect library not installed (optional for subtitle language detection)")
            logger.warning("Install with: pip install langdetect")
            logger.warning("Subtitle language detection will use fallback method")
    
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
        logger.info(f"Scanning directories: {config.path}")
    else:
        print(f"Scanning directories: {config.path}")
        
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