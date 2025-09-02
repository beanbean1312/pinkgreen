import streamlit as st
from PIL import Image, ImageEnhance, ImageChops, ImageFilter
import numpy as np
import io

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

st.set_page_config(layout="centered", page_title="Brave Pink & Hero Green Editor")

FILTER_PRESETS = {
    "Hero Green & Brave Pink": {"shadow": [27, 96, 47], "highlight": [247, 132, 197]},
    "Ocean Deep": {"shadow": [13, 40, 79], "highlight": [125, 206, 214]},
    "Sunset": {"shadow": [61, 28, 92], "highlight": [245, 149, 87]},
    "Noir": {"shadow": [0, 0, 0], "highlight": [240, 240, 240]},
    "Crimson & Gold": {"shadow": [105, 24, 24], "highlight": [227, 183, 96]}
}

OVERLAY_FILES = {
    "None": None,
    "Light Leak 1": "light_leak_1.png", 
    "Light Leak 2": "light_leak_2.png", 
    "Light Leak 3": "light_leak_3.png",
    "Dust 1": "dust_1.png", 
    "Dust 2": "dust_2.png", 
    "Scratches": "scratches_1.png"
}

MAX_DIMENSION = 3000

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def hex_to_rgb(hex_color):
    """Convert hex color string to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def calculate_dimensions(width, height):
    """Calculate new dimensions while maintaining aspect ratio within MAX_DIMENSION."""
    if width <= MAX_DIMENSION and height <= MAX_DIMENSION:
        return width, height
    
    scale = MAX_DIMENSION / max(width, height)
    return int(width * scale), int(height * scale)

# =============================================================================
# IMAGE PROCESSING FUNCTIONS
# =============================================================================

def apply_clarity(image, strength):
    """Apply clarity enhancement using unsharp mask."""
    clarity_percent = int(strength * 2.5)
    return image.filter(ImageFilter.UnsharpMask(radius=5, percent=clarity_percent, threshold=3))


def apply_shadow_cleanup(image, threshold):
    """Clean up shadows by setting pixels below threshold to black."""
    img_array = np.array(image.convert('RGB'))
    luminance = (0.2126 * img_array[:, :, 0] + 
                 0.7152 * img_array[:, :, 1] + 
                 0.0722 * img_array[:, :, 2])
    
    shadow_mask = luminance < threshold
    img_array[shadow_mask] = [0, 0, 0]
    return Image.fromarray(img_array.astype(np.uint8))


def apply_split_toning(image, controls):
    """Apply split toning effect with separate shadow and highlight tints."""
    shadow_tint_rgb = np.array(hex_to_rgb(controls["shadow_tint"]))
    highlight_tint_rgb = np.array(hex_to_rgb(controls["highlight_tint"]))
    strength = controls["split_toning_strength"] / 100.0
    
    img_array = np.array(image.convert('RGB'), dtype=np.float32)
    luminance = (0.2126 * img_array[:, :, 0] + 
                 0.7152 * img_array[:, :, 1] + 
                 0.0722 * img_array[:, :, 2]) / 255.0
    
    shadow_mask = (1.0 - luminance)[:, :, np.newaxis]
    highlight_mask = luminance[:, :, np.newaxis]
    
    shadow_layer = shadow_tint_rgb[np.newaxis, np.newaxis, :]
    highlight_layer = highlight_tint_rgb[np.newaxis, np.newaxis, :]
    
    final_array = (img_array * (1.0 - shadow_mask * strength) + 
                   shadow_layer * shadow_mask * strength)
    final_array = (final_array * (1.0 - highlight_mask * strength) + 
                   highlight_layer * highlight_mask * strength)
    
    return Image.fromarray(np.clip(final_array, 0, 255).astype(np.uint8))


def apply_duotone_filter(image, controls):
    """Apply duotone filter effect using preset color palettes."""
    palette = FILTER_PRESETS[controls["duotone_filter_name"]]
    
    if controls["invert_duotone"]:
        shadow_rgb = np.array(palette["highlight"])
        highlight_rgb = np.array(palette["shadow"])
    else:
        shadow_rgb = np.array(palette["shadow"])
        highlight_rgb = np.array(palette["highlight"])
    
    rgb_array = np.array(image.convert('RGB'))
    luminance = (0.2126 * rgb_array[:, :, 0] + 
                 0.7152 * rgb_array[:, :, 1] + 
                 0.0722 * rgb_array[:, :, 2])
    
    min_lum, max_lum = luminance.min(), luminance.max()
    lum_range = max_lum - min_lum or 1
    norm_lum = (luminance - min_lum) / lum_range
    
    exponent = controls["duotone_contrast"]
    enhanced_lum = np.where(
        norm_lum < 0.5,
        (np.power(norm_lum * 2, exponent) / 2),
        1 - (np.power((1 - norm_lum) * 2, exponent) / 2)
    )
    
    final_array = shadow_rgb + enhanced_lum[:, :, np.newaxis] * (highlight_rgb - shadow_rgb)
    return Image.fromarray(np.clip(final_array, 0, 255).astype(np.uint8))


def apply_vignette(image, strength):
    """Apply vignette effect that darkens image edges."""
    img_array = np.array(image.convert('RGB'))
    height, width = img_array.shape[:2]
    
    x, y = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
    radius = np.sqrt(x**2 + y**2) / np.sqrt(2)
    
    vignette_strength = strength / 100.0 * 2.5
    mask = (1 - radius**2) ** vignette_strength
    
    img_array = (img_array * mask[:, :, np.newaxis]).astype(np.uint8)
    return Image.fromarray(img_array)


def apply_film_grain(image, strength):
    """Add film grain noise to the image."""
    img_array = np.array(image.convert('RGB'), dtype=np.float32)
    noise_intensity = (strength / 100.0) * 25
    noise = np.random.normal(0, noise_intensity, img_array.shape)
    
    noisy_array = img_array + noise
    noisy_array = np.clip(noisy_array, 0, 255)
    return Image.fromarray(noisy_array.astype(np.uint8))


def apply_overlay(base_image, overlay_style, strength):
    """Apply texture overlay using screen blend mode."""
    try:
        overlay_filename = OVERLAY_FILES[overlay_style]
        overlay_path = f"overlays/{overlay_filename}"
        overlay_image = Image.open(overlay_path).convert("RGB")
    except (FileNotFoundError, TypeError):
        st.warning(f"Could not load overlay file: {overlay_style}. Make sure it exists in the 'overlays' folder.")
        return base_image
    
    base_image_rgb = base_image.convert("RGB")
    overlay_image = overlay_image.resize(base_image_rgb.size)
    screen_blended = ImageChops.screen(base_image_rgb, overlay_image)
    
    alpha = strength / 100.0
    final_image = Image.blend(base_image_rgb, screen_blended, alpha)
    return final_image


def apply_all_effects(image, controls):
    """Apply all selected effects to the image in order."""
    processed_image = image.copy()
    
    # Resize if necessary
    new_width, new_height = calculate_dimensions(processed_image.width, processed_image.height)
    if (new_width, new_height) != (processed_image.width, processed_image.height):
        processed_image = processed_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Basic adjustments
    enhancer = ImageEnhance.Brightness(processed_image)
    processed_image = enhancer.enhance(1.0 + controls["brightness"] / 100.0)
    
    enhancer = ImageEnhance.Color(processed_image)
    processed_image = enhancer.enhance(controls["saturation"] / 100.0)
    
    # Enhancement effects
    if controls["clarity"] > 0:
        processed_image = apply_clarity(processed_image, controls["clarity"])
    
    if controls["shadow_cleanup"] > 0:
        processed_image = apply_shadow_cleanup(processed_image, controls["shadow_cleanup"])
    
    # Color grading effects
    if controls["enable_split_toning"]:
        processed_image = apply_split_toning(processed_image, controls)
    
    if controls["duotone_filter_name"] != "None":
        processed_image = apply_duotone_filter(processed_image, controls)
    
    # Stylistic effects
    if controls["vignette"] > 0:
        processed_image = apply_vignette(processed_image, controls["vignette"])
    
    if controls["overlay_style"] != "None":
        processed_image = apply_overlay(processed_image, controls["overlay_style"], controls["overlay_strength"])
    
    if controls["film_grain"] > 0:
        processed_image = apply_film_grain(processed_image, controls["film_grain"])
    
    return processed_image

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def initialize_session_state():
    """Initialize all control values in session state."""
    if 'controls_initialized' not in st.session_state:
        defaults = {
            "duotone_filter_name": "Hero Green & Brave Pink",
            "duotone_contrast": 1.8,
            "invert_duotone": False,
            "brightness": 0,
            "saturation": 100,
            "clarity": 0,
            "shadow_cleanup": 0,
            "enable_split_toning": False,
            "shadow_tint": "#0000FF",
            "highlight_tint": "#FFA500",
            "split_toning_strength": 50,
            "vignette": 0,
            "film_grain": 0,
            "overlay_style": "None",
            "overlay_strength": 75,
            "h_flip": False,
            "v_flip": False,
            "controls_initialized": True
        }
        st.session_state.update(defaults)

# =============================================================================
# RESET FUNCTIONS
# =============================================================================

def reset_duotone_controls():
    """Reset duotone filter controls to defaults."""
    st.session_state.update({
        "duotone_filter_name": "Hero Green & Brave Pink",
        "duotone_contrast": 1.8,
        "invert_duotone": False
    })


def reset_image_adjustments():
    """Reset basic image adjustment controls to defaults."""
    st.session_state.update({
        "brightness": 0,
        "saturation": 100,
        "shadow_cleanup": 0,
        "clarity": 0
    })


def reset_advanced_color_grading():
    """Reset color grading controls to defaults."""
    st.session_state.update({
        "enable_split_toning": False,
        "shadow_tint": "#0000FF",
        "highlight_tint": "#FFA500",
        "split_toning_strength": 50
    })


def reset_stylistic_effects():
    """Reset stylistic effect controls to defaults."""
    st.session_state.update({
        "vignette": 0,
        "film_grain": 0,
        "overlay_style": "None",
        "overlay_strength": 75
    })

# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_header():
    """Render the app header with title and subtitle."""
    st.markdown("""
        <style>
            .title { text-align: center; }
            .subtitle { text-align: center; }
            .footer { text-align: center; color: #888; }
            .download-button-container {
                display: flex;
                justify-content: center;
            }
        </style>
        <h1 class="title">Brave Pink & Hero Green Editor</h1>
        <p class="subtitle">Transform your photos with beautiful duotone effects. Local & private.</p>
        """, 
        unsafe_allow_html=True
    )


def render_sidebar_controls():
    """Render all sidebar control panels and return control values."""
    st.sidebar.header("Controls")
    controls = {}
    
    # Duotone Filter Controls
    with st.sidebar.expander("Duotone Filter", expanded=True):
        controls["duotone_filter_name"] = st.selectbox(
            "Filter Preset", 
            ["None"] + list(FILTER_PRESETS.keys()), 
            key="duotone_filter_name"
        )
        controls["duotone_contrast"] = st.slider(
            "Filter Contrast", 1.0, 4.0, 
            key="duotone_contrast", step=0.1
        )
        controls["invert_duotone"] = st.checkbox(
            "Invert Filter Colors", 
            key="invert_duotone"
        )
        st.button("Reset Duotone", on_click=reset_duotone_controls, use_container_width=True)
    
    # Image Adjustments
    with st.sidebar.expander("Image Adjustments"):
        controls["brightness"] = st.slider("Brightness", -100, 100, key="brightness")
        controls["saturation"] = st.slider("Saturation", 0, 200, key="saturation")
        controls["clarity"] = st.slider(
            "Clarity", 0, 100, key="clarity", 
            help="Enhances texture and detail in mid-tones."
        )
        controls["shadow_cleanup"] = st.slider(
            "Shadow Cleanup", 0, 255, key="shadow_cleanup", 
            help="Crushes dark, noisy areas to a solid color."
        )
        st.button("Reset Adjustments", on_click=reset_image_adjustments, use_container_width=True)
    
    # Advanced Color Grading
    with st.sidebar.expander("Advanced Color Grading"):
        controls["enable_split_toning"] = st.checkbox(
            "Enable Split Toning", 
            key="enable_split_toning"
        )
        controls["shadow_tint"] = st.color_picker("Shadow Tint", key="shadow_tint")
        controls["highlight_tint"] = st.color_picker("Highlight Tint", key="highlight_tint")
        controls["split_toning_strength"] = st.slider(
            "Strength", 0, 100, 
            key="split_toning_strength"
        )
        st.button("Reset Color Grading", on_click=reset_advanced_color_grading, use_container_width=True)
    
    # Stylistic Effects
    with st.sidebar.expander("Stylistic Effects"):
        controls["vignette"] = st.slider("Vignette Strength", 0, 100, key="vignette")
        controls["film_grain"] = st.slider(
            "Film Grain", 0, 100, key="film_grain", 
            help="Adds a vintage, textured look."
        )
        st.sidebar.markdown("---")
        controls["overlay_style"] = st.selectbox(
            "Overlay Style", 
            options=list(OVERLAY_FILES.keys()), 
            key="overlay_style"
        )
        controls["overlay_strength"] = st.slider("Overlay Strength", 0, 100, key="overlay_strength")
        st.button("Reset Stylistic", on_click=reset_stylistic_effects, use_container_width=True)
    
    return controls


def render_transform_controls():
    """Render transform control buttons and return button states."""
    st.sidebar.subheader("Transform")
    col1, col2 = st.sidebar.columns(2)
    
    flip_horizontal = col1.button("Flip Horizontal", use_container_width=True)
    flip_vertical = col2.button("Flip Vertical", use_container_width=True)
    
    return flip_horizontal, flip_vertical


def render_download_button(processed_image):
    """Render centered download button for processed image."""
    buf = io.BytesIO()
    processed_image.save(buf, format="PNG")
    byte_im = buf.getvalue()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.download_button(
            label="Download Edited Image", 
            data=byte_im, 
            file_name="edited_image.png", 
            mime="image/png",
            use_container_width=True
        )


def render_footer():
    """Render the app footer."""
    st.markdown("---")
    st.markdown("""
        <p class="footer">
            Your photos are processed privately and are never stored.<br>
            Developed by <strong>Alex</strong>
        </p>
        """, 
        unsafe_allow_html=True
    )

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Render UI components
    render_header()
    controls = render_sidebar_controls()
    flip_horizontal, flip_vertical = render_transform_controls()
    
    # File upload
    uploaded_file = st.file_uploader(
        "Drop your image here or click to browse", 
        type=["jpg", "jpeg", "png", "webp"]
    )
    
    if uploaded_file is not None:
        # Handle flip transformations
        if flip_horizontal:
            st.session_state.h_flip = not st.session_state.h_flip
        if flip_vertical:
            st.session_state.v_flip = not st.session_state.v_flip
        
        # Load and apply transformations
        original_image = Image.open(uploaded_file)
        if st.session_state.h_flip:
            original_image = original_image.transpose(Image.FLIP_LEFT_RIGHT)
        if st.session_state.v_flip:
            original_image = original_image.transpose(Image.FLIP_TOP_BOTTOM)
        
        # Update controls with session state values
        for key in st.session_state:
            if key in controls or hasattr(st.session_state, key):
                controls[key] = st.session_state[key]
        
        # Process image and display
        processed_image = apply_all_effects(original_image, controls)
        st.image(processed_image, caption="Your Edited Image", use_container_width=True)
        
        # Render download button
        render_download_button(processed_image)
        
    else:
        # Reset flip states when no image is uploaded
        st.session_state.h_flip = False
        st.session_state.v_flip = False
        st.info("Upload an image to begin editing.")
    
    # Render footer
    render_footer()


if __name__ == "__main__":
    main()