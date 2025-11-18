from openai import OpenAI
from openai import RateLimitError, APIError, AuthenticationError
from pydub import AudioSegment
from dotenv import load_dotenv
import os
import time
import sys

# Load environment variables from .env file
load_dotenv()

# Check if API key is set
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("ERROR: OPENAI_API_KEY not found in environment variables.")
    print("Please set it in your .env file or as an environment variable:")
    print("  export OPENAI_API_KEY='your-api-key-here'")
    sys.exit(1)

# Initialize OpenAI client with API key from environment
client = OpenAI(api_key=api_key)


def tts(text, voice, filename, max_retries=3):
    """
    Generate text-to-speech audio with retry logic for rate limits.

    Args:
        text: Text to convert to speech
        voice: Voice name (e.g., 'alloy', 'verse')
        filename: Output filename
        max_retries: Maximum number of retries for rate limit errors
    """
    for attempt in range(max_retries):
        try:
            with client.audio.speech.with_streaming_response.create(
                model="gpt-4o-mini-tts", voice=voice, input=text
            ) as response:
                response.stream_to_file(filename)
            print(f"✓ Generated: {filename}")
            return True
        except RateLimitError as e:
            wait_time = (attempt + 1) * 5  # Exponential backoff: 5s, 10s, 15s
            if attempt < max_retries - 1:
                print(
                    f"⚠ Rate limit hit. Waiting {wait_time} seconds before retry {attempt + 2}/{max_retries}..."
                )
                time.sleep(wait_time)
            else:
                print(f"❌ ERROR: Rate limit exceeded after {max_retries} attempts.")
                print(f"   File: {filename}")
                print(f"   Error: {e}")
                print("\n   Solutions:")
                print(
                    "   1. Check your OpenAI account quota at https://platform.openai.com/usage"
                )
                print("   2. Add billing information if needed")
                print("   3. Wait a few minutes and try again")
                return False
        except AuthenticationError as e:
            print(f"❌ ERROR: Authentication failed for {filename}")
            print(f"   Error: {e}")
            print("   Please check your OPENAI_API_KEY is valid")
            return False
        except APIError as e:
            print(f"❌ ERROR: API error for {filename}")
            print(f"   Error: {e}")
            return False
        except Exception as e:
            print(f"❌ ERROR: Unexpected error for {filename}")
            print(f"   Error: {e}")
            return False

    return False


# Create individual audio lines
print("Generating audio files...\n")
audio_files = [
    ("Hey Jamie, long time no see! How've you been?", "alloy", "line1_male.mp3"),
    (
        "Hey! I've been good. Just busy with work and trying to keep a healthier routine. How about you?",
        "verse",
        "line2_female.mp3",
    ),
    (
        "Same here. I finally started waking up earlier. I've been taking morning walks around the neighborhood.",
        "alloy",
        "line3_male.mp3",
    ),
    (
        "Really? That sounds nice. I keep telling myself I'll start exercising, but I always end up staying on the couch after work.",
        "verse",
        "line4_female.mp3",
    ),
    (
        "I get that. I used to be the same. Starting small helped—just 10 minutes a day.",
        "alloy",
        "line5_male.mp3",
    ),
    (
        "That actually sounds doable. Maybe I'll try a short walk after dinner tonight.",
        "verse",
        "line6_female.mp3",
    ),
    (
        "Do it! It really clears your head. And you sleep better too.",
        "alloy",
        "line7_male.mp3",
    ),
    (
        "True. Oh! Have you tried that new coffee place on Maple Street?",
        "verse",
        "line8_female.mp3",
    ),
    ("Not yet. Is it good?", "alloy", "line9_male.mp3"),
    (
        "Yeah, super cozy. We should go this weekend. Could be my reward for starting the new routine.",
        "verse",
        "line10_female.mp3",
    ),
    (
        "Deal! A walk and then coffee. Sounds like the perfect plan.",
        "alloy",
        "line11_male.mp3",
    ),
]

successful = 0
failed = 0
skipped = 0
for i, (text, voice, filename) in enumerate(audio_files, 1):
    # Skip if file already exists
    if os.path.exists(filename):
        print(f"[{i}/{len(audio_files)}] Skipping {filename} (already exists)")
        skipped += 1
        successful += 1  # Count as successful since we can use it
        continue

    print(f"[{i}/{len(audio_files)}] Generating {filename}...")
    if tts(text, voice, filename):
        successful += 1
    else:
        failed += 1
        print(f"   Skipping {filename} due to error\n")

print(
    f"\n✓ Generated {successful} files successfully ({skipped} already existed, {successful - skipped} newly generated)"
)
if failed > 0:
    print(f"✗ Failed to generate {failed} files")
    print(
        "\nNote: You can run this script again - it will skip files that already exist."
    )

    # If all files failed, exit with error
    if successful == 0:
        print("\n❌ ERROR: No files were generated. Please check:")
        print("   1. Your OpenAI API key is valid")
        print("   2. You have sufficient quota/billing set up")
        print("   3. Check https://platform.openai.com/usage for your account status")
        sys.exit(1)
# Combine all audio files
print("\nCombining audio files into dialogue_full.mp3...")
final = AudioSegment.empty()

for i in range(1, 12):
    filename = f"line{i}_{'male' if i % 2 == 1 else 'female'}.mp3"
    if os.path.exists(filename):
        final += AudioSegment.from_mp3(filename) + AudioSegment.silent(500)
    else:
        print(f"⚠ Warning: {filename} not found, skipping...")

if len(final) > 0:
    final.export("dialogue_full.mp3", format="mp3")
    print(f"✓ Created dialogue_full.mp3 ({len(final)/1000:.1f} seconds)")
else:
    print("❌ ERROR: No audio files found to combine")
    sys.exit(1)
