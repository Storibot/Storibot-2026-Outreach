"""
Storibot Video Shot Generator using Runway Gen-4 Turbo
Generates cinematic video shots from text prompts.
"""

import os
import time
import requests
from pathlib import Path

# Runway API configuration
RUNWAY_API_KEY = os.getenv("RUNWAY_API_KEY")
if not RUNWAY_API_KEY:
    raise ValueError("RUNWAY_API_KEY environment variable not set!")

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

    8: """Clean, premium end card composition: Centered robot logo on subtle
dark gradient. Soft teal glow emanates from logo. Minimal, confident,
tech-premium aesthetic. Slight camera push-in for energy. Space for
text overlays.""",
}


def generate_video_shot(shot_number: int, prompt: str) -> Path:
    """Generate a single video shot using Runway Gen-4 Turbo"""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"shot_{shot_number:02d}.mp4"

    print(f"\n{'='*60}")
    print(f"Generating Shot {shot_number:02d}")
    print(f"{'='*60}")
    print(f"Prompt: {prompt[:100]}...")

    # Runway Gen-4 Turbo API endpoint
    url = "https://api.dev.runwayml.com/v1/text_to_video"

    headers = {
        "Authorization": f"Bearer {RUNWAY_API_KEY}",
        "Content-Type": "application/json",
        "X-Runway-Version": "2024-11-06"
    }

    payload = {
        "model": "gen4_turbo",
        "promptText": prompt,
        "duration": 5,
        "ratio": "16:9"
    }

    # Start video generation
    print("Starting video generation...")
    response = requests.post(url, json=payload, headers=headers)

    if response.status_code not in [200, 201]:
        print(f"Error starting generation: {response.status_code}")
        print(response.text)
        raise Exception(f"Failed to start video generation: {response.text}")

    result = response.json()
    task_id = result.get("id")

    if not task_id:
        raise Exception(f"No task ID returned: {result}")

    print(f"Task started: {task_id}")

    # Poll for completion
    poll_url = f"https://api.dev.runwayml.com/v1/tasks/{task_id}"

    max_attempts = 120  # 10 minutes max (video gen can take a while)
    for attempt in range(max_attempts):
        time.sleep(5)
        print(f"Checking status... (attempt {attempt + 1}/{max_attempts})")

        poll_response = requests.get(poll_url, headers=headers)
        if poll_response.status_code != 200:
            print(f"Poll error: {poll_response.status_code}")
            continue

        status = poll_response.json()
        task_status = status.get("status")

        print(f"Status: {task_status}")

        if task_status == "SUCCEEDED":
            print("Generation complete!")

            # Get the video URL
            output_urls = status.get("output", [])
            if output_urls:
                video_url = output_urls[0]

                # Download the video
                print(f"Downloading video...")
                video_response = requests.get(video_url)

                with open(output_file, "wb") as f:
                    f.write(video_response.content)

                print(f"Saved to: {output_file}")
                print(f"File size: {len(video_response.content) / 1024:.1f} KB")
                return output_file

            raise Exception("No output URL in response")

        elif task_status == "FAILED":
            error = status.get("error", "Unknown error")
            raise Exception(f"Generation failed: {error}")

        elif task_status in ["PENDING", "RUNNING"]:
            continue

    raise Exception("Video generation timed out")


def generate_all_shots():
    """Generate all video shots"""
    print("Storibot Video Shot Generator (Runway Gen-4 Turbo)")
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


def generate_single_shot(shot_number: int):
    """Generate a single shot by number"""
    if shot_number not in SHOT_PROMPTS:
        print(f"Shot {shot_number} not found. Available: {list(SHOT_PROMPTS.keys())}")
        return None

    return generate_video_shot(shot_number, SHOT_PROMPTS[shot_number])


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Generate specific shot
        shot_num = int(sys.argv[1])
        generate_single_shot(shot_num)
    else:
        # Generate all shots
        generate_all_shots()
