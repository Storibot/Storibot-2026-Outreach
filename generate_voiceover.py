"""
Storibot Voiceover Generator using ElevenLabs API
Generates professional voiceover audio from script text.
"""

import os
import requests
from pathlib import Path

# ElevenLabs API configuration
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
if not ELEVENLABS_API_KEY:
    raise ValueError("ELEVENLABS_API_KEY environment variable not set!")

# Voice settings
VOICE_NAME = "Adam"  # Deep, authoritative voice
VOICE_ID = "pNInz6obpgDQGcFmaJgB"  # Adam's voice ID

# Audio generation settings (calm, professional tone)
VOICE_SETTINGS = {
    "stability": 0.5,
    "similarity_boost": 0.75,
    "style": 0.3
}

# Output configuration
OUTPUT_DIR = Path("audio/voiceover")
OUTPUT_FILE = OUTPUT_DIR / "storibot_vo.wav"

# Voiceover script
VOICEOVER_SCRIPT = """The AI revolution created infinite content. And infinite noise. But stories? Real stories that move people? Those still take craft. Storibot dot A I. Hollywood storytelling frameworks. Powered by AI. Transform any idea into a captivating narrative. In seconds. The future of AI storytelling. Start your free trial today."""


def generate_voiceover():
    """Generate voiceover audio using ElevenLabs API"""

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ElevenLabs text-to-speech endpoint
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"

    headers = {
        "Accept": "audio/wav",
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY
    }

    payload = {
        "text": VOICEOVER_SCRIPT,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": VOICE_SETTINGS,
        "output_format": "pcm_48000"  # 48kHz PCM for WAV conversion
    }

    print(f"Generating voiceover with voice: {VOICE_NAME}")
    print(f"Settings: stability={VOICE_SETTINGS['stability']}, "
          f"similarity_boost={VOICE_SETTINGS['similarity_boost']}, "
          f"style={VOICE_SETTINGS['style']}")

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code != 200:
        raise Exception(f"ElevenLabs API error: {response.status_code} - {response.text}")

    # Get raw PCM audio data
    pcm_data = response.content

    # Convert PCM to WAV (48kHz, 16-bit, mono)
    wav_data = create_wav_header(pcm_data, sample_rate=48000, bits_per_sample=16, channels=1)

    # Save to file
    with open(OUTPUT_FILE, "wb") as f:
        f.write(wav_data)

    print(f"Voiceover saved to: {OUTPUT_FILE}")
    print(f"Format: 48kHz WAV mono")
    print(f"File size: {len(wav_data) / 1024:.1f} KB")

    return OUTPUT_FILE


def create_wav_header(pcm_data: bytes, sample_rate: int = 48000,
                      bits_per_sample: int = 16, channels: int = 1) -> bytes:
    """Create WAV file with proper header from raw PCM data"""
    import struct

    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    data_size = len(pcm_data)

    # WAV header structure
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',                    # ChunkID
        36 + data_size,             # ChunkSize
        b'WAVE',                    # Format
        b'fmt ',                    # Subchunk1ID
        16,                         # Subchunk1Size (PCM)
        1,                          # AudioFormat (1 = PCM)
        channels,                   # NumChannels
        sample_rate,                # SampleRate
        byte_rate,                  # ByteRate
        block_align,                # BlockAlign
        bits_per_sample,            # BitsPerSample
        b'data',                    # Subchunk2ID
        data_size                   # Subchunk2Size
    )

    return header + pcm_data


if __name__ == "__main__":
    try:
        output_path = generate_voiceover()
        print(f"\nSuccess! Voiceover generated at: {output_path}")
    except Exception as e:
        print(f"Error generating voiceover: {e}")
        raise
