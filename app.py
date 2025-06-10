import streamlit as st
import openai
import base64
from PIL import Image
from io import BytesIO
import time
import os

# Page configuration
st.set_page_config(
    page_title="Auralis",
    #page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 2rem;
    }
    
    .main-title {
        font-size: 4rem;
        font-weight: 600;
        color: #333;
        margin-bottom: 0.5rem;
    }
    
    .main-subtitle {
        font-size: 1.5rem;
        font-weight: 400;
        color: #666;
        margin-bottom: 2rem;
    }
    
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
    }
    
    .stChatMessage {
        background-color: #f8f9fa;
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .stChatMessage[data-testid="user-message"] {
        background-color: #e3f2fd;
    }
    
    .stChatMessage[data-testid="assistant-message"] {
        background-color: #f1f8e9;
    }
    
    .generated-image {
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .error-message {
        color: #d32f2f;
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #d32f2f;
    }
    
    .success-message {
        color: #388e3c;
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #388e3c;
    }
    
    .generating-message {
        color: #f57c00;
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #f57c00;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = """
You are Auralis, a helpful and funny AI assistant who gives detailed responses. You can engage in conversations on a 
wide range of topics and help users with various tasks. Be friendly and informative in your responses.
"""

# Keywords that trigger image generation
IMAGE_KEYWORDS = ['create', 'visual', 'image', 'generate', 'picture', 'draw', 'make', 'design', 'art', 'illustration']

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "openai_client" not in st.session_state:
    # Initialize OpenAI client
    try:
        # Try to get API key from Streamlit secrets first, then environment
        api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("⚠️ OpenAI API key not found. Please set it in Streamlit secrets or environment variables.")
            st.stop()
        
        st.session_state.openai_client = openai.OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize OpenAI client: {str(e)}")
        st.stop()

def generate_image(prompt):
    """Generate image using DALL-E and return PIL Image object"""
    try:
        # Clean and validate the prompt
        clean_prompt = prompt.strip()
        if len(clean_prompt) > 1000:
            clean_prompt = clean_prompt[:1000]
        
        # Add delay to avoid rate limiting
        time.sleep(1)
        
        image_response = st.session_state.openai_client.images.generate(
            model="dall-e-3",
            prompt=clean_prompt,
            size="1024x1024",
            n=1,
            response_format="b64_json",
            quality="standard"
        )
        
        # Process the image
        image_base64 = image_response.data[0].b64_json
        image_data = base64.b64decode(image_base64)
        
        # Convert to PIL Image
        image = Image.open(BytesIO(image_data))
        return image, None
        
    except openai.RateLimitError as e:
        return None, "Rate limit exceeded. Please wait a moment before generating another image."
    except openai.APIError as e:
        return None, f"OpenAI API error: {e}"
    except Exception as e:
        return None, f"Unexpected error generating image: {str(e)}"

def generate_text_response(message, messages_history):
    """Generate streaming text response using OpenAI API"""
    try:
        # Prepare messages for API
        api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # Add conversation history
        for msg in messages_history:
            if msg["role"] in ["user", "assistant"]:
                # Only add text content, skip images
                if "content" in msg and isinstance(msg["content"], str):
                    api_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
        
        # Add current message
        api_messages.append({"role": "user", "content": message})
        
        # Create streaming response
        stream = st.session_state.openai_client.chat.completions.create(
            model=MODEL,
            messages=api_messages,
            stream=True
        )
        
        # Stream the response
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
                
    except Exception as e:
        yield f"Sorry, I encountered an error: {str(e)}"

def should_generate_image(message):
    """Check if message should trigger image generation"""
    return any(keyword in message.lower() for keyword in IMAGE_KEYWORDS)

def is_valid_image_prompt(message):
    """Validate if the message is suitable for image generation"""
    clean_message = message.strip()
    
    # Check minimum length
    if len(clean_message) < 10:
        return False, "Please provide a more detailed description (at least 10 characters) to generate a high-quality image."
    
    # Check for forbidden content
    forbidden_words = ['nsfw', 'nude', 'explicit', 'violence', 'blood', 'gore']
    if any(word in clean_message.lower() for word in forbidden_words):
        return False, "I can't generate images with that content. Please try a different, more appropriate prompt."
    
    return True, None

# Header
st.markdown("""
<div class="main-header">
    <h1 class="main-title">Auralis</h1>
    <h2 class="main-subtitle">What are you working on?</h2>
</div>
""", unsafe_allow_html=True)

# Sidebar with controls
with st.sidebar:
    st.title("Controls")
    
    if st.button("🗑️ Clear Chat", type="secondary", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    st.subheader("💡 Tips")
    st.markdown("""
    - Use words like **'create'**, **'generate'**, **'draw'** for images
    - Be specific with your image descriptions
    - Ask questions about any topic
    - I can help with coding, writing, analysis, and more!
    """)
    
    st.divider()
    
    st.subheader("🎯 Example Prompts")
    example_prompts = [
        "What is artificial intelligence?",
        "Create a beautiful sunset over mountains",
        "Explain quantum computing simply",
        "Generate a futuristic city image",
        "Benefits of renewable energy?",
        "Draw a cute robot helping humans"
    ]
    
    for prompt in example_prompts:
        if st.button(prompt, key=f"example_{hash(prompt)}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.rerun()

# Main chat container
with st.container():
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.write(message["content"])
            else:
                # Assistant message
                if "content" in message:
                    st.write(message["content"])
                
                # Display image if present
                if "image" in message:
                    st.image(message["image"], caption="Generated Image", use_column_width=True)

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Process the message
    if should_generate_image(prompt):
        # Image generation flow
        is_valid, error_msg = is_valid_image_prompt(prompt)
        
        if not is_valid:
            # Show error message
            with st.chat_message("assistant"):
                st.markdown(f'<div class="error-message">{error_msg}</div>', unsafe_allow_html=True)
            
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        
        else:
            # Show generating message
            with st.chat_message("assistant"):
                generating_placeholder = st.empty()
                generating_placeholder.markdown(
                    f'<div class="generating-message">🎨 Generating image for: "{prompt}"<br>This may take a few moments...</div>', 
                    unsafe_allow_html=True
                )
                
                # Generate image
                image, error = generate_image(prompt)
                
                # Clear generating message
                generating_placeholder.empty()
                
                if error:
                    # Show error with tips
                    error_with_tips = error
                    if "rate limit" in error.lower():
                        error_with_tips += "\n\n💡 Tip: Try waiting 30-60 seconds before generating another image."
                    elif "server" in error.lower():
                        error_with_tips += "\n\n💡 Tip: OpenAI servers might be busy. Try again in a few minutes."
                    else:
                        error_with_tips += "\n\n💡 Tip: Try rephrasing your prompt or making it more specific."
                    
                    st.markdown(f'<div class="error-message">{error_with_tips}</div>', unsafe_allow_html=True)
                    st.session_state.messages.append({"role": "assistant", "content": error_with_tips})
                
                else:
                    # Show success message and image
                    success_msg = f"✨ Here's your generated image: '{prompt}'"
                    st.markdown(f'<div class="success-message">{success_msg}</div>', unsafe_allow_html=True)
                    st.image(image, caption="Generated Image", use_column_width=True)
                    
                    # Add to chat history with image
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": success_msg,
                        "image": image
                    })
    
    else:
        # Text generation flow
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Stream the response
            for response_chunk in generate_text_response(prompt, st.session_state.messages):
                full_response += response_chunk
                message_placeholder.write(full_response + "▌")
            
            # Final response without cursor
            message_placeholder.write(full_response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p><em>Tip: Use descriptive prompts for better image generation results!</em></p>
    </div>
    """, 
    unsafe_allow_html=True
)
