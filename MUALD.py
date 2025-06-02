import os
import sys
import subprocess
import tempfile
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import whisper
import logging
import shutil
import yaml
import warnings
import time
import requests
from packaging import version
import psutil
import threading

VERSION = '1.3'

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
        self.path = "."
        self.remux_to_mkv = False
        self.show_details = True
        self.whisper_model = "base"
        self.dry_run = False
        
    def load_from_file(self, config_path: str = "config.yml"):
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
                
            logger.info(f"Configuration loaded from {config_path}")
            
        except Exception as e:
            logger.error(f"Error loading config file {config_path}: {e}")
            logger.info("Using default configuration")
    
    def create_sample_config(self, config_path: str = "config.yml"):
        sample_config = {
            'path': 'P:\Movies',
            'remux_to_mkv': True,
            'show_details': False,
            'whisper_model': 'base',
            'dry_run': False
        }
        
        try:
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
        
        if not config.show_details:
            warnings.filterwarnings("ignore", 
                                  message="FP16 is not supported on CPU; using FP32 instead",
                                  module="whisper.transcribe")
        
        self.whisper_model = whisper.load_model(config.whisper_model)
        
        self.ffmpeg = find_executable('ffmpeg')
        self.ffprobe = find_executable('ffprobe') 
        self.mkvpropedit = find_executable('mkvpropedit')
        
        if not all([self.ffmpeg, self.ffprobe, self.mkvpropedit]):
            missing = []
            if not self.ffmpeg: missing.append('ffmpeg')
            if not self.ffprobe: missing.append('ffprobe')
            if not self.mkvpropedit: missing.append('mkvpropedit')
            raise RuntimeError(f"Missing executables: {', '.join(missing)}")
        
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
        
        # Video file extensions to consider for remuxing
        self.video_extensions = {'.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.m2ts', '.mts', '.ts', '.vob'}
    
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
                    import time
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
                                time.sleep(1.0 * (attempt + 1))  # Increasing delay
                            else:
                                raise e
                                
                except Exception as e:
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
                language = tags.get('language', '').lower()
                
                if not language or language in ['und', 'unknown', 'undefined']:
                    undefined_tracks.append((audio_track_count, stream, i))
                    if self.config.show_details:
                        logger.info(f"Found undefined audio track {audio_track_count} (stream {i}) in {file_path.name}")
                
                audio_track_count += 1
        
        return undefined_tracks
    
    def extract_audio_sample(self, file_path: Path, audio_track_index: int, stream_index: int,
                           duration: int = 45, start_time: int = 180) -> Optional[Path]:
        try:
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_path = Path(temp_file.name)
            temp_file.close()
            
            mapping_strategies = [
                f'0:a:{audio_track_index}',  # Audio track index
                f'0:{stream_index}',         # Stream index
                f'a:{audio_track_index}',    # Audio only
            ]
            
            # Try multiple time segments to avoid intros/music/silence
            time_segments = [
                (start_time, duration),      # Original: 3 minutes in, 45 seconds
                (120, duration),             # 2 minutes in, 45 seconds  
                (300, duration),             # 5 minutes in, 45 seconds
                (60, duration),              # 1 minute in, 45 seconds
                (0, duration)                # From beginning
            ]
            
            for segment_start, segment_duration in time_segments:
                for i, map_strategy in enumerate(mapping_strategies):
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
                        
                        # Apply CPU limiting to subprocess
                        limited_cmd = limit_subprocess_resources(cmd)
                        
                        if self.config.show_details:
                            logger.debug(f"Trying segment {segment_start}s with mapping {map_strategy}")
                        
                        result = subprocess.run(limited_cmd, check=True, capture_output=True, 
                                              text=True, encoding='utf-8', errors='replace')
                        
                        if temp_path.exists() and temp_path.stat().st_size > 5000:  # At least 5KB for better quality
                            try:
                                validate_cmd = [
                                    self.ffprobe, '-v', 'quiet', '-show_entries', 
                                    'stream=duration,bit_rate', '-of', 'csv=p=0', str(temp_path)
                                ]
                                validate_result = subprocess.run(validate_cmd, capture_output=True, text=True, check=True, encoding='utf-8', errors='replace')
                                
                                # Also check for volume level to avoid silent segments
                                volume_cmd = [
                                    self.ffmpeg, '-i', str(temp_path), '-af', 'volumedetect', 
                                    '-f', 'null', '-', '-v', 'quiet', '-stats'
                                ]
                                volume_result = subprocess.run(volume_cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
                                
                                # Look for reasonable volume levels (not silence)
                                if 'mean_volume:' in volume_result.stderr:
                                    # Extract mean volume (typically negative dB value)
                                    for line in volume_result.stderr.split('\n'):
                                        if 'mean_volume:' in line:
                                            try:
                                                volume_db = float(line.split('mean_volume:')[1].split('dB')[0].strip())
                                                # Reject if too quiet (likely silence)
                                                if volume_db < -50:  # Very quiet threshold
                                                    if self.config.show_details:
                                                        logger.debug(f"Rejecting segment due to low volume: {volume_db}dB")
                                                    continue
                                            except (ValueError, IndexError):
                                                pass
                                
                                if self.config.show_details:
                                    logger.info(f"Successfully extracted quality audio from {segment_start}s using {map_strategy}")
                                return temp_path
                                
                            except subprocess.CalledProcessError:
                                # If validation fails, still use the file if it exists and has reasonable size
                                if temp_path.stat().st_size > 5000:
                                    if self.config.show_details:
                                        logger.info(f"Extracted audio (validation skipped) from {segment_start}s using {map_strategy}")
                                    return temp_path
                        else:
                            if self.config.show_details:
                                logger.debug(f"Audio sample too small from segment {segment_start}s")
                            
                    except subprocess.CalledProcessError as e:
                        if self.config.show_details:
                            error_msg = e.stderr if hasattr(e, 'stderr') and e.stderr else str(e)
                            logger.debug(f"Segment {segment_start}s with mapping {map_strategy} failed: {error_msg}")
                        continue
                        
                    if temp_path.exists():
                        temp_path.unlink()
                        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                        temp_path = Path(temp_file.name)
                        temp_file.close()
            
            if temp_path.exists():
                temp_path.unlink()
            logger.error("All extraction strategies and time segments failed")
            return None
            
        except Exception as e:
            logger.error(f"Unexpected error during audio extraction: {e}")
            if 'temp_path' in locals() and temp_path.exists():
                temp_path.unlink()
            return None
    
    def detect_language(self, audio_path: Path) -> Optional[str]:
        try:
            if not audio_path.exists() or audio_path.stat().st_size < 1000:
                logger.error(f"Audio file is too small or doesn't exist: {audio_path}")
                return None
            
            if self.config.show_details:
                logger.info(f"Analyzing audio file: {audio_path} ({audio_path.stat().st_size} bytes)")
            
            transcribe_options = {
                "language": None,  # Let Whisper auto-detect
                "task": "transcribe",
                "temperature": 0.0,
                "best_of": 3,
                "beam_size": 5,
                "patience": 1.0,
                "length_penalty": 1.0,
                "suppress_tokens": "-1",
                "initial_prompt": None,
                "condition_on_previous_text": True,
                "fp16": False,
                "compression_ratio_threshold": 2.4,
                "logprob_threshold": -1.0,
                "no_speech_threshold": 0.6
            }
            
            # Try to add VAD filter options if supported by the Whisper version
            try:
                # Test if VAD filter is supported by trying to create DecodingOptions with it
                import whisper.decoding
                test_options = whisper.decoding.DecodingOptions(
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500)
                )
                # If no exception, VAD is supported
                transcribe_options["vad_filter"] = True
                transcribe_options["vad_parameters"] = dict(min_silence_duration_ms=500)
                if self.config.show_details:
                    logger.info("Using VAD filter for voice activity detection")
            except (TypeError, AttributeError):
                # VAD filter not supported in this Whisper version
                if self.config.show_details:
                    logger.info("VAD filter not available in this Whisper version, using standard detection")
            
            result = self.whisper_model.transcribe(str(audio_path), **transcribe_options)
            
            detected_language = result['language']
            segments = result.get('segments', [])
            confidence = result.get('language_probability', 0)
            
            if segments:
                segment_confidences = []
                for segment in segments:
                    if 'avg_logprob' in segment:
                        segment_conf = min(1.0, max(0.0, (segment['avg_logprob'] + 1.0)))
                        segment_confidences.append(segment_conf)
                
                if segment_confidences:
                    avg_confidence = sum(segment_confidences) / len(segment_confidences)
                    confidence = max(confidence, avg_confidence)
            
            text_sample = result.get('text', '').strip()
            
            if self.config.show_details:
                logger.info(f"Detected language: {detected_language} (confidence: {confidence:.2f})")
                logger.info(f"Sample text: '{text_sample[:100]}'")
            
            text_length = len(text_sample)
            word_count = len(text_sample.split()) if text_sample else 0
            
            # Enhanced silent content detection
            # Check if segments have high no_speech_prob (indicating silence/music)
            segments_with_speech = []
            if segments:
                segments_with_speech = [s for s in segments if s.get('no_speech_prob', 1.0) < 0.5]
            
            is_silent = (
                # Original silence detection
                ((detected_language.lower() == 'english' or detected_language.lower() == 'en') and
                 confidence < 0.1 and text_length == 0 and word_count == 0) or
                # Segment-based silence detection (works with or without VAD)
                (len(segments) > 0 and len(segments_with_speech) == 0 and text_length < 10) or
                # General low-content detection
                (confidence < 0.05 and text_length < 5) or
                # Hallucination detection - repetitive or nonsensical characters
                (text_length > 0 and self._is_likely_hallucination(text_sample))
            )
            
            if is_silent:
                if self.config.show_details:
                    logger.info("No speech detected or likely hallucination - marking as 'no linguistic content' (zxx)")
                return 'zxx'
            
            # Adjust confidence thresholds based on text quality
            min_confidence = 0.25 if text_length > 20 else 0.15
            
            if (confidence > min_confidence and text_length > 5 and word_count > 1) or confidence > 0.5:
                iso_code = self.language_codes.get(detected_language.lower(), detected_language)
                
                if detected_language.lower() == 'dutch':
                    iso_code = 'dut'
                elif detected_language.lower() == 'nl':
                    iso_code = 'dut'
                
                return iso_code
            else:
                if self.config.show_details:
                    logger.warning(f"Low confidence ({confidence:.2f}) or insufficient text (length: {text_length}, words: {word_count})")
                return None
            
        except Exception as e:
            logger.error(f"Error detecting language for {audio_path}: {e}")
            return None
        finally:
            if audio_path.exists():
                audio_path.unlink()

    def _is_likely_hallucination(self, text: str) -> bool:
        """
        Detect if the transcribed text is likely a hallucination from silent audio.
        
        Args:
            text: The transcribed text to analyze
            
        Returns:
            True if the text appears to be hallucinated, False otherwise
        """
        if not text or len(text.strip()) == 0:
            return True
        
        text = text.strip()
        
        # Check for repetitive single characters or very short sequences
        if len(set(text.replace(' ', ''))) <= 3 and len(text) > 10:
            return True
        
        # Check for non-Latin scripts that might indicate hallucination
        # (especially common with silent audio)
        non_latin_count = 0
        for char in text:
            if ord(char) > 127:  # Non-ASCII
                # Check if it's from scripts commonly hallucinated on silent audio
                if (0x1780 <= ord(char) <= 0x17FF or  # Khmer
                    0x0E00 <= ord(char) <= 0x0E7F or  # Thai
                    0x1000 <= ord(char) <= 0x109F or  # Myanmar
                    0x0980 <= ord(char) <= 0x09FF):   # Bengali
                    non_latin_count += 1
        
        # If more than 50% of characters are from potentially hallucinated scripts
        if len(text) > 0 and (non_latin_count / len(text)) > 0.5:
            return True
        
        # Check for very repetitive patterns
        words = text.split()
        if len(words) > 3:
            unique_words = set(words)
            if len(unique_words) / len(words) < 0.3:  # Less than 30% unique words
                return True
        
        # Check for sequences of identical short strings
        if len(text) > 20:
            # Look for patterns like "្្ ្្ ្្" or similar
            parts = text.split()
            if len(parts) > 5:
                unique_parts = set(parts)
                if len(unique_parts) <= 2:  # Only 1-2 unique "words" repeated
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
        
        # Find undefined audio tracks
        undefined_tracks = self.find_undefined_audio_tracks(mkv_path)
        results['undefined_tracks'] = len(undefined_tracks)
        
        if not undefined_tracks:
            if self.config.show_details:
                logger.info(f"No undefined audio tracks found in {mkv_path.name}")
            return results
        
        # Process each undefined track
        for audio_track_index, stream_info, stream_index in undefined_tracks:
            try:
                # Extract audio sample
                audio_sample = self.extract_audio_sample(mkv_path, audio_track_index, stream_index)
                if not audio_sample:
                    results['failed_tracks'].append(audio_track_index)
                    results['errors'].append(f"Failed to extract audio from track {audio_track_index}")
                    continue
                
                # Detect language
                language_code = self.detect_language(audio_sample)
                if not language_code:
                    results['failed_tracks'].append(audio_track_index)
                    results['errors'].append(f"Failed to detect language for track {audio_track_index}")
                    continue
                
                # Update metadata
                success = self.update_mkv_language(mkv_path, audio_track_index, language_code, self.config.dry_run)
                if success:
                    results['processed_tracks'].append({
                        'track_index': audio_track_index,
                        'detected_language': language_code
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
                print("🔄 UPDATE AVAILABLE")
                print(f"{'='*60}")
                print(f"Current version: {current_version}")
                print(f"Latest version:  {latest_version}")
                print("Download from: https://github.com/netplexflix/MKV-Undefined-Audio-Language-Detector")
                print(f"{'='*60}\n")
            else:
                print("✓ Up to date")
                
        except Exception as e:
            if latest_version != current_version:
                print("Update may be available")
                print(f"Current: {current_version}, Latest: {latest_version}")
                print("Check: https://github.com/netplexflix/MKV-Undefined-Audio-Language-Detector\n")
            else:
                print("✓ Up to date")
        
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

def print_detailed_summary(results: List[Dict], config: Config, runtime_seconds: float):
    print(f"\n{'='*60}")
    print("PROCESSING SUMMARY")
    print(f"{'='*60}")
    
    total_files = len(results)
    files_with_actions = 0
    files_processed = 0
    silent_content_files = []
    
    # ANSI color codes
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    
    for result in results:
        file_name = Path(result['original_file']).name
        actions = []
        has_silent_content = False
        
        if result['was_remuxed']:
            actions.append("remuxed")
        
        processed_tracks = sorted(result['processed_tracks'], key=lambda x: x['track_index'])
        failed_tracks = sorted(result.get('failed_tracks', []))
        
        track_actions = []
        
        for track in processed_tracks:
            track_idx = track['track_index']
            lang_code = track['detected_language']
            if lang_code == 'zxx':
                track_actions.append(f"{YELLOW}track{track_idx}: {lang_code} (no speech){RESET}")
                has_silent_content = True
            else:
                track_actions.append(f"track{track_idx}: {lang_code}")
        
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
        
        if show_file:
            files_with_actions += 1
    
    if files_with_actions > 0:
        print(f"\nShowing {files_with_actions} files that required action (out of {total_files} total files)")
        if files_processed > 0:
            print(f"Successfully processed {files_processed} files")
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
    
    print(f"\nTotal runtime: {format_duration(runtime_seconds)}")
    
    if config.dry_run:
        print("(Dry run - no files were actually modified)")
    print()

def main():
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description='Detect and update language metadata for video file audio tracks')
    parser.add_argument('--config', default='config.yml', 
                       help='Configuration file path (default: config.yml)')
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
    
    # Set final logging level based on configuration (after loading config)
    if config.show_details:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)
       
    # Validate directory
    if not os.path.isdir(config.path):
        logger.error(f"Directory not found: {config.path}")
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
    
    # Initialize detector
    try:
        if config.show_details:
            logger.info(f"Loading Whisper model: {config.whisper_model}")
        else:
            print(f"Loading Whisper model: {config.whisper_model}")
        detector = MKVLanguageDetector(config)
    except RuntimeError as e:
        logger.error(f"Failed to initialize detector: {e}")
        sys.exit(1)
    
    # Process directory
    if config.show_details:
        logger.info(f"Scanning directory: {config.path}")
    else:
        print(f"Scanning directory: {config.path}")
        
    if config.dry_run:
        print("DRY RUN MODE - No files will be modified")
    
    results = detector.process_directory(config.path)
    
    end_time = time.time()
    runtime_seconds = end_time - start_time
    
    print_detailed_summary(results, config, runtime_seconds)

if __name__ == "__main__":
    main()