import gradio as gr
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from PIL import Image
import os
import io
import base64
from gtts import gTTS
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# --- CONFIG ---
TEXT_MODEL = "gemini-2.5-flash"
IMAGE_MODEL = "gemini-2.5-flash-image"

SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

SYSTEM_PROMPT = """You are a magical children's picture book creator.
Create ONE page at a time in a warm, age 4–8 friendly style.

Output EXACTLY this format:
PAGE {page_number}: [Short catchy title]
STORY:
2–4 simple, engaging sentences. Use vivid action and wonder.
IMAGE_PROMPT:
A detailed visual description for illustration (art style: whimsical watercolor, soft colors)."""

# --- HELPERS ---

def text_to_speech(text):
    try:
        clean_text = text.replace('#', '').replace('*', '').strip()
        tts = gTTS(text=clean_text, lang='en', slow=False)
        audio_path = "story_audio.mp3"
        tts.save(audio_path)
        return audio_path
    except: return None

def generate_image_gemini(prompt, reference_image=None):
    try:
        model = genai.GenerativeModel(model_name=IMAGE_MODEL, safety_settings=SAFETY_SETTINGS)
        full_prompt = (prompt + ", children's picture book illustration, watercolor style, cute, whimsical, no text.")
        content = [full_prompt]
        if reference_image:
            content.insert(0, reference_image)
            content[0] = "Maintain character consistency from this reference: " + full_prompt

        response = model.generate_content(content)
        for part in response.parts:
            if hasattr(part, 'inline_data'):
                return Image.open(io.BytesIO(base64.b64decode(part.inline_data.data)))
        return None
    except Exception as e:
        if "429" in str(e) or "quota" in str(e).lower(): return "quota_error"
        return None

def get_page_count_text(current, total):
    if total == 0: return ""
    return f"✨ **Page {current + 1} of {total}** ✨"

# --- CORE LOGIC ---

def finish_story(history_state):
    if not history_state:
        raise gr.Error("You haven't started a story yet!")
    
    # Calculate how many pages were made
    stories_count = sum(1 for x in history_state if isinstance(x, str) and "STORY:" in x)
    
    final_text = f"""
    # 🌟 THE END 🌟
    
    **What a wonderful adventure!** You created a beautiful book with {stories_count} pages. 
    
    Hope to see you again soon for another magical story! ✨
    """
    
    # Generate a celebratory audio
    audio_path = text_to_speech("The End. What a wonderful adventure! Thanks for reading.")
    
    # We return a 'None' image or a special 'The End' graphic if you have one
    return None, final_text, audio_path, stories_count

def navigate_pages(direction, current_idx, history):
    if not history: return None, "### No pages yet!", None, 0, ""
    
    # Filter for story blocks (every story block starts with "PAGE")
    stories = [i for i, item in enumerate(history) if isinstance(item, str) and "STORY:" in item]
    total_pages = len(stories)
    
    new_idx = max(0, min(current_idx + direction, total_pages - 1))
    story_raw = history[stories[new_idx]]
    
    try:
        story_text = story_raw.split("STORY:")[1].split("IMAGE_PROMPT:")[0].strip()
    except:
        story_text = story_raw
        
    img_out = None
    # The image is usually saved right after the text block in history
    if stories[new_idx] + 1 < len(history) and isinstance(history[stories[new_idx] + 1], Image.Image):
        img_out = history[stories[new_idx] + 1]
    
    audio_path = text_to_speech(story_text)
    display_markdown = f"## {story_raw.split('STORY:')[0].strip()}\n\n{story_text}"
    return img_out, display_markdown, audio_path, new_idx, get_page_count_text(new_idx, total_pages)

def generate_story_step(user_data, history_state):
    user_text = (user_data["text"] or "").strip()
    uploaded_files = user_data.get("files", [])

    if not user_text and not uploaded_files:
        # Instead of returning a string, we raise a clear Error modal
        raise gr.Error("Please enter an idea to keep the magic going! ✨")

    model = genai.GenerativeModel(model_name=TEXT_MODEL, system_instruction=SYSTEM_PROMPT, safety_settings=SAFETY_SETTINGS)
    
    # Build context
    messages = []
    for i, item in enumerate(history_state):
        if isinstance(item, str):
            role = "model" if "STORY:" in item else "user"
            messages.append({"role": role, "parts": [item]})

    current_input = [user_text] if user_text else ["Create the next page of our story."]
    if uploaded_files:
        current_input.append(Image.open(uploaded_files[0]))

    messages.append({"role": "user", "parts": current_input})

    try:
        response = model.generate_content(messages)
        raw_text = response.text
        
        # Parsing Logic
        try:
            story = raw_text.split("STORY:")[1].split("IMAGE_PROMPT:")[0].strip()
            image_prompt = raw_text.split("IMAGE_PROMPT:")[1].strip()
            page_title = raw_text.split("STORY:")[0].strip()
        except:
            story = raw_text
            image_prompt = raw_text
            page_title = "The Next Chapter"

        # Image Generation
        prev_img = next((item for item in reversed(history_state) if isinstance(item, Image.Image)), None)
        img_out = generate_image_gemini(image_prompt, prev_img)
        audio_path = text_to_speech(story)
        
        # Update History
        history_state.append(user_text or "[Magic Wand]")
        history_state.append(raw_text)
        if isinstance(img_out, Image.Image):
            history_state.append(img_out)

        stories_count = sum(1 for x in history_state if isinstance(x, str) and "STORY:" in x)
        new_idx = stories_count - 1
        
        display_text = f"## {page_title}\n\n{story}"
        
        # Handle Image Quota specifically
        if img_out == "quota_error":
            gr.Warning("The artist is taking a nap! (Image limit reached). We'll keep the story going with text.")
            img_out = None
            
        return img_out, display_text, audio_path, history_state, {"text": "", "files": []}, new_idx, get_page_count_text(new_idx, stories_count)

    except Exception as e:
        # THIS IS THE MAGIC PART:
        error_msg = str(e).lower()
        if "429" in error_msg or "quota" in error_msg or "exhausted" in error_msg:
            raise gr.Error("🌟 The Magic Book is resting for today (Daily Limit Reached). Try again tomorrow!")
        else:
            raise gr.Error(f"Oops! A forest sprite caused an error: {str(e)}")

# --- CUSTOM UI ---

kids_theme = gr.themes.Soft(
    primary_hue="pink",
    secondary_hue="yellow",
    font=[gr.themes.GoogleFont("Comic Neue"), "sans-serif"]
)

custom_css = """
    .finish-btn {
        background: #E1BEE7 !important; /* Light Lavender */
        color: #7B1FA2 !important;
        border: 2px solid #7B1FA2 !important;
    }
    .clear-btn {
        border-radius: 50px !important;
        border: 2px solid #D1C4E9 !important; /* Soft Purple */
        background: white !important;
        transition: all 0.3s ease !important;
        box-shadow: 4px 4px 0px #D1C4E9 !important;
        color: #9575CD !important;
        font-weight: bold !important;
    }
    
    .clear-btn:hover {
        transform: scale(1.05);
        background: #F3E5F5 !important;
        box-shadow: 2px 2px 0px #D1C4E9 !important;
    }
    .gradio-container { background: linear-gradient(180deg, #FFF0F5 0%, #E0F7FA 100%); }
    .book-frame { 
        background: white; border: 6px solid #FFB6C1; border-radius: 30px; 
        padding: 20px; box-shadow: 15px 15px 0px #FFB6C1; 
    }
    .page-counter { 
        text-align: center; color: #FF69B4; font-size: 1.1em; 
        font-weight: bold; background: #FFF0F5; border-radius: 12px; padding: 4px;
    }
    .nav-btn {
        border-radius: 50px !important; border: 2px solid #FFB6C1 !important;
        background: white !important; transition: all 0.3s ease !important;
        box-shadow: 4px 4px 0px #FFB6C1 !important; color: #FF69B4 !important;
    }
    .nav-btn:hover { transform: scale(1.1) rotate(-2deg); background: #FFF0F5 !important; }
    .magic-btn {
        background: linear-gradient(90deg, #FF69B4, #FFB6C1) !important;
        color: white !important; border-radius: 20px !important; font-size: 1.3em !important;
        box-shadow: 0px 6px 0px #D05090 !important;
    }
    .magic-btn:active { transform: translateY(3px); box-shadow: 0px 2px 0px #D05090 !important; }
"""

with gr.Blocks(theme=kids_theme, css=custom_css) as demo:
    gr.Markdown("# 📖✨ PlotPix: My Magical Picture Book ✨📖")
    
    history_state = gr.State([])
    current_page_idx = gr.State(0)

    with gr.Row(elem_classes="book-frame"):
        with gr.Column(scale=1):
            main_image = gr.Image(label="Illustration", interactive=False, height=450)
        with gr.Column(scale=1):
            story_display = gr.Markdown("### Welcome, Little Author!\nDescribe your adventure to begin.")
            audio_player = gr.Audio(label="🔊 Listen!", autoplay=True)
            
            with gr.Column():
                page_counter = gr.Markdown("", elem_classes="page-counter")
                with gr.Row():
                    prev_btn = gr.Button("👈 Back", elem_classes="nav-btn")
                    next_btn = gr.Button("Next 👉", elem_classes="nav-btn")

    msg = gr.MultimodalTextbox(label="What happens next?", submit_btn=False, lines=2)
    
    with gr.Row():
        submit_btn = gr.Button("✨ Turn the Page ✨", variant="primary", size="lg", elem_classes="magic-btn")
        clear_btn = gr.Button("🔄 New Story", variant="secondary",elem_classes="clear-btn")
        finish_btn = gr.Button("🏰 The End", variant="secondary", elem_classes="clear-btn")
    # --- EVENTS ---
    finish_btn.click(
        finish_story,
        [history_state],
        [main_image, story_display, audio_player, current_page_idx]
    )

    submit_btn.click(
        generate_story_step, 
        [msg, history_state], 
        [main_image, story_display, audio_player, history_state, msg, current_page_idx, page_counter]
    )
    
    prev_btn.click(
        lambda idx, hist: navigate_pages(-1, idx, hist), 
        [current_page_idx, history_state], 
        [main_image, story_display, audio_player, current_page_idx, page_counter]
    )
    
    next_btn.click(
        lambda idx, hist: navigate_pages(1, idx, hist), 
        [current_page_idx, history_state], 
        [main_image, story_display, audio_player, current_page_idx, page_counter]
    )
    
    clear_btn.click(
        lambda: (None, "### Let's start a new adventure!", None, [], 0, ""), 
        None, 
        [main_image, story_display, audio_player, history_state, current_page_idx, page_counter]
    )

if __name__ == "__main__":
    demo.launch(debug=True)