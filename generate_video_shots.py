"""
Storibot Video Shot Generator using Google Veo 3
Generates cinematic video shots from text prompts.
"""

import os
import time
import requests
from pathlib import Path

# Google API configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set!")

# Output configuration
OUTPUT_DIR = Path("video/shots")

# Video shot prompts
SHOT_PROMPTS = {
    1: """Cinematic aerial shot pushing through an endless storm of floating
digital content — tweets, posts, articles, thumbnails — swirling
chaotically in a dark void. Cold blue lighting. Shallow depth of field.
4K film grain. Camera slowly pushes through the noise.""",

    2: """Extreme close-up of a human eye reflecting chaotic scrolling content.
The pupil dilates. Overwhelmed expression. Cool desaturated color grade.
Cinematic lighting with subtle teal rim light. Slow motion.""",

    3: """Hands of a master craftsman carefully shaping light itself into a
glowing narrative thread. Warm golden particles form into elegant
story structure. Dark workshop environment. Dramatic shadows.
Reverent, artisanal feel. Anamorphic lens flares.""",

    4: """Cinematic reveal: A minimal robot face logo emerges from darkness,
rendered in elegant teal light (#2DD4BF). Particles coalesce around it.
Clean, futuristic environment with subtle grid lines. Premium tech
aesthetic. Logo rotates slowly revealing depth.""",

    5: """Abstract visualization of the Hero's Journey narrative structure as a
luminous 3D pathway. Glowing nodes represent story beats. Camera flies
through the structure. Teal (#2DD4BF) and white against deep black.
Data visualization meets cinematic storytelling.""",

    6: """Neural network visualization pulsing with creative energy. Synapses
fire in teal light. Camera pulls back to reveal the network forms
the shape of a human brain merged with circuitry. Clean, sophisticated
look. Subtle particle effects.""",

    7: """Timelapse-style shot: A simple text prompt transforms into a fully
formed narrative document. Words materialize, organize themselves,
and polish into professional content. Elegant UI elements. Teal
highlights on dark interface. Satisfying visual transformation.""",
}


def generate_video_shot(shot_number: int, prompt: str) -> Path:
    """Generate a single video shot using Google Veo 3"""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"shot_{shot_number:02d}.mp4"

    print(f"\n{'='*60}")
    print(f"Generating Shot {shot_number:02d}")
    print(f"{'='*60}")
    print(f"Prompt: {prompt[:100]}...")

    # Google Veo 3 API endpoint (via Generative AI API)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/veo-2.0-generate-001:predictLongRunning?key={GOOGLE_API_KEY}"

    payload = {
        "instances": [{
            "prompt": prompt
        }],
        "parameters": {
            "aspectRatio": "16:9",
            "personGeneration": "allow_adult",
            "durationSeconds": 5,
            "numberOfVideos": 1
        }
    }

    headers = {"Content-Type": "application/json"}

    # Start video generation (long-running operation)
    print("Starting video generation...")
    response = requests.post(url, json=payload, headers=headers)

    if response.status_code != 200:
        print(f"Error starting generation: {response.status_code}")
        print(response.text)
        raise Exception(f"Failed to start video generation: {response.text}")

    operation = response.json()
    operation_name = operation.get("name")

    if not operation_name:
        raise Exception(f"No operation name returned: {operation}")

    print(f"Operation started: {operation_name}")

    # Poll for completion
    poll_url = f"https://generativelanguage.googleapis.com/v1beta/{operation_name}?key={GOOGLE_API_KEY}"

    max_attempts = 60  # 5 minutes max
    for attempt in range(max_attempts):
        time.sleep(5)
        print(f"Checking status... (attempt {attempt + 1}/{max_attempts})")

        poll_response = requests.get(poll_url)
        if poll_response.status_code != 200:
            print(f"Poll error: {poll_response.status_code}")
            continue

        result = poll_response.json()

        if result.get("done"):
            print("Generation complete!")

            # Extract video data
            if "response" in result:
                videos = result["response"].get("predictions", [])
                if videos:
                    video_data = videos[0].get("video", {})
                    video_uri = video_data.get("uri")

                    if video_uri:
                        # Download the video
                        print(f"Downloading video...")
                        video_response = requests.get(video_uri)

                        with open(output_file, "wb") as f:
                            f.write(video_response.content)

                        print(f"Saved to: {output_file}")
                        print(f"File size: {len(video_response.content) / 1024:.1f} KB")
                        return output_file

            # Check for errors
            if "error" in result:
                raise Exception(f"Generation failed: {result['error']}")

            raise Exception(f"Unexpected response format: {result}")

    raise Exception("Video generation timed out")


def generate_all_shots():
    """Generate all video shots"""
    print("Storibot Video Shot Generator")
    print(f"Generating {len(SHOT_PROMPTS)} shots...")

    generated = []
    for shot_num, prompt in sorted(SHOT_PROMPTS.items()):
        try:
            output_path = generate_video_shot(shot_num, prompt)
            generated.append(output_path)
        except Exception as e:
            print(f"Failed to generate shot {shot_num}: {e}")

    print(f"\n{'='*60}")
    print(f"Generated {len(generated)}/{len(SHOT_PROMPTS)} shots")
    for path in generated:
        print(f"  - {path}")

    return generated


if __name__ == "__main__":
    generate_all_shots()
