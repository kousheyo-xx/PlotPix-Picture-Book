import gradio as gr
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from PIL import Image
import os
import io
import base64
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

TEXT_MODEL = "gemini-2.5-flash"          # Text/story generation (or try "gemini-1.5-flash-8b" if available)
IMAGE_MODEL = "gemini-2.5-flash-image"   # Nano Banana – fast image gen
# For higher quality (if you have access/billing): IMAGE_MODEL = "gemini-3-pro-image-preview"

SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

system_prompt = """You are a magical children's picture book creator.
Create ONE page at a time in a warm, age 4–8 friendly style.

Output EXACTLY this format:

PAGE {page_number}: [Short catchy title – max 6 words]

STORY:
2–4 simple, engaging sentences. Use vivid action, emotions, repetition and wonder.

IMAGE_PROMPT:
A detailed visual description for illustration (2-3 sentences describing the scene, characters, mood, and art style)."""

def generate_image_gemini(prompt, reference_image=None):
    """Generate image using Gemini Nano Banana"""
    try:
        model = genai.GenerativeModel(
            model_name=IMAGE_MODEL,
            safety_settings=SAFETY_SETTINGS
        )

        full_prompt = (
            prompt + 
            ", children's picture book illustration, watercolor style, cute, colorful, whimsical, storybook art, "
            "soft lighting, no text in image, aspect ratio 3:4 for portrait book page"
        )

        content = [full_prompt]

        if reference_image:
            content.insert(0, reference_image)
            content[0] = (
                "Maintain consistent character designs, colors, style, mood, and overall aesthetic from the reference image. "
                + full_prompt
            )

        response = model.generate_content(content)

        for part in response.parts:
            if hasattr(part, 'inline_data') and part.inline_data.mime_type.startswith('image/'):
                image_data = part.inline_data.data
                image_bytes = base64.b64decode(image_data)
                img = Image.open(io.BytesIO(image_bytes))
                print("Gemini Nano Banana image generated successfully!")
                return img

        print("No image data in response")
        return None

    except Exception as e:
        error_str = str(e)
        if "429" in error_str or "quota" in error_str.lower():
            print("Quota exceeded in image generation")
            return "quota_error"
        print(f"Gemini image error: {error_str}")
        return None

def generate_story_step(user_data, history):
    user_text = (user_data["text"] or "").strip()
    uploaded_files = user_data.get("files", [])

    if not user_text and not uploaded_files and history:
        user_text = "Continue the story with the next page."

    if not user_text and not uploaded_files:
        return history, {"text": "", "files": []}

    text_model = genai.GenerativeModel(
        model_name=TEXT_MODEL,
        system_instruction=system_prompt,
        safety_settings=SAFETY_SETTINGS
    )

    messages = []
    for msg in history:
        if isinstance(msg["content"], str):
            role = "user" if msg["role"] == "user" else "model"
            messages.append({"role": role, "parts": [msg["content"]]})
        # Skip images in text history

    current_parts = [user_text] if user_text else ["Create the next page."]

    if uploaded_files:
        for path in uploaded_files[:2]:
            try:
                img = Image.open(path)
                current_parts.append(img)
            except:
                pass
        if any(isinstance(p, Image.Image) for p in current_parts):
            current_parts.append("Keep consistent character design, colors and mood from reference image(s).")

    messages.append({"role": "user", "parts": current_parts})

    try:
        response = text_model.generate_content(messages)
        text = response.text.strip() if response.text else "No response."

        lines = text.split("\n")
        page_title = f"Page {len(history)//2 + 1}"
        story = ""
        image_prompt = ""

        current_section = None
        for line in lines:
            line = line.strip()
            if line.startswith("PAGE"):
                page_title = line
            elif line.startswith("STORY:") or line == "STORY":
                current_section = "story"
            elif line.startswith("IMAGE_PROMPT:") or line == "IMAGE_PROMPT":
                current_section = "image"
            elif current_section == "story" and line and not line.startswith("IMAGE"):
                story += line + " "
            elif current_section == "image" and line:
                image_prompt += line + " "

        story = story.strip() or "A beautiful moment..."
        image_prompt = image_prompt.strip()

        final_prompt = image_prompt if image_prompt else story[:150]

        user_display = user_text or "[Reference image(s) uploaded]"
        history.append({"role": "user", "content": user_display})

        story_content = f"### {page_title}\n\n{story}\n\n🎨 *Generating illustration...*"
        history.append({"role": "assistant", "content": story_content})

        yield history, {"text": "", "files": []}

        previous_img = None
        for msg in reversed(history[:-1]):
            if isinstance(msg.get("content"), Image.Image):
                previous_img = msg["content"]
                break

        generated_image = generate_image_gemini(final_prompt, previous_img)

        if generated_image == "quota_error":
            history[-1]["content"] = (
                f"### {page_title}\n\n{story}\n\n"
                "⚠️ **Quota limit reached for image generation.**\n"
                "Your free daily requests are used up. Wait until tomorrow (resets daily)"
                f"**Manual prompt you can use right now in gemini.google.com (select 'Create images'):**\n{final_prompt}"
            )
        elif generated_image:
            history[-1]["content"] = f"### {page_title}\n\n{story}"
            history.append({"role": "assistant", "content": generated_image})
        else:
            history[-1]["content"] = (
                f"### {page_title}\n\n{story}\n\n"
                f"❌ *Image generation failed (check console for details).*\n"
                f"**Try this prompt manually in Gemini web:** {final_prompt}"
            )

        yield history, {"text": "", "files": []}

    except Exception as e:
        error_str = str(e)
        if "429" in error_str or "quota" in error_str.lower():
            msg = (
                "⚠️ **Quota exceeded!** Your free tier daily limit (likely 20 requests) is reached for today.\n"
                "Wait until tomorrow for reset."
            )
        else:
            msg = f"❌ Error: {error_str}\n\nTry again or check API key / model access!"
        history.append({"role": "assistant", "content": msg})
        yield history, {"text": "", "files": []}

with gr.Blocks() as demo:
    gr.Markdown("""
    # 📖✨ PlotPix Picture Book Creator 📖✨
    
    **Powered by Gemini**
    
    Create magical children's stories with AI-generated text AND illustrations!
    Describe what happens next (or upload refs) — all powered by Gemini.
    
    ⏱️ *Note: Free tier has daily limits (~20 requests/day)*
    """)

    chatbot = gr.Chatbot(
        label="Your Picture Book",
        height=600
    )

    with gr.Row():
        msg = gr.MultimodalTextbox(
            label="What happens next?",
            placeholder="The little fox finds a glowing crystal cave...",
            file_types=["image"],
            file_count="multiple",
            lines=2,
        )
    
    with gr.Row():
        submit_btn = gr.Button("📝 Generate Next Page", variant="primary", size="lg")
        clear_btn = gr.Button("🔄 Start New Story", variant="secondary")

    gr.Markdown("""
    💡 **Tip:** Upload a character image on page 1 for better consistency across pages!  
    If you hit quota limits, wait 24h.
    """)

    submit_btn.click(
        generate_story_step,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg]
    )
    
    msg.submit(
        generate_story_step,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg]
    )

    clear_btn.click(
        lambda: ([], {"text":"", "files":[]}),
        None,
        [chatbot, msg]
    )

if __name__ == "__main__":
    demo.launch(
        server_port=7860,
        show_error=True,
        share=True,
        theme=gr.themes.Soft()
    )