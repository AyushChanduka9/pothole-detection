import os
import cv2
import gradio as gr
import shutil
import uuid
import subprocess

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "processed"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def detect_potholes_image(input_image):
    """Process image and return before/after preview"""
    if input_image is None:
        return None, None, "No image uploaded"
    
    try:
        # Generate unique filename
        base = str(uuid.uuid4())
        input_path = os.path.join(UPLOAD_DIR, base + ".jpg")
        output_path = os.path.join(OUTPUT_DIR, base + "_result.jpg")
        
        # Save input image
        cv2.imwrite(input_path, cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR))
        
        # Load YOLO model
        with open(os.path.join("project_files", 'obj.names'), 'r') as f:
            classes = f.read().splitlines()
        net = cv2.dnn.readNet('project_files/yolov4_tiny.weights',
                              'project_files/yolov4_tiny.cfg')
        model = cv2.dnn_DetectionModel(net)
        model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)
        
        # Detect potholes
        img = cv2.imread(input_path)
        classIds, scores, boxes = model.detect(img, confThreshold=0.6, nmsThreshold=0.4)
        
        pothole_count = len(classIds)
        
        # Draw bounding boxes
        for (classId, score, box) in zip(classIds, scores, boxes):
            cv2.rectangle(img, (box[0], box[1]),
                          (box[0] + box[2], box[1] + box[3]),
                          color=(0, 255, 0), thickness=2)
            label = f"Pothole: {score:.2f}"
            cv2.putText(img, label, (box[0], box[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imwrite(output_path, img)
        
        # Convert for Gradio display
        output_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        info = f"Detection Complete! Found {pothole_count} pothole(s)"
        return input_image, output_img, info
        
    except Exception as e:
        return None, None, f"Error processing image: {str(e)}"

def detect_potholes_video(input_video):
    """Process video and return output video path"""
    if input_video is None:
        return None, "No video uploaded"
    
    try:
        # Generate unique filename
        base = str(uuid.uuid4())
        input_path = os.path.join(UPLOAD_DIR, base + ".mp4")
        output_path = os.path.join(OUTPUT_DIR, base + "_result.avi")
        
        # Copy input video
        shutil.copy(input_video, input_path)
        
        # Use existing test.mp4 for processing
        test_path = "test.mp4"
        shutil.copy(input_path, test_path)
        
        # Run detection
        subprocess.run(['python', 'camera_video.py'], check=True)
        
        if os.path.exists("result.avi"):
            shutil.move("result.avi", output_path)
            return output_path, "Video processing complete!"
        else:
            return None, "Error: Video processing failed."
            
    except Exception as e:
        return None, f"Error processing video: {str(e)}"

# Custom CSS for modern, dynamic UI with dark theme
custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background: #000000 !important;
    }
    .main-header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.5);
        margin-bottom: 2rem;
        color: white;
    }
    .upload-section {
        background: #1a1a1a;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.3);
    }
    .result-section {
        background: #1a1a1a;
        padding: 1.5rem;
        border-radius: 15px;
        margin-top: 1rem;
    }
    .stats-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
"""

# Create Gradio interface with enhanced UI
with gr.Blocks(css=custom_css, title="AI Pothole Detection System", theme=gr.themes.Soft()) as demo:
    
    # Main Header
    with gr.Column(elem_classes="main-header"):
        gr.Markdown("""
        # 🚗 AI-Powered Pothole Detection System
        ### Detect road potholes instantly using advanced YOLOv4-Tiny AI model
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("🎯 **Accurate Detection**")
            with gr.Column(scale=1):
                gr.Markdown("⚡ **Fast Processing**")
            with gr.Column(scale=1):
                gr.Markdown("📊 **Detailed Results**")
    
    # Tabs with enhanced content
    with gr.Tab("📸 Image Detection"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Upload Image")
                image_input = gr.Image(
                    label="Drop your image here or click to upload", 
                    type="numpy",
                    height=400
                )
                with gr.Row():
                    image_button = gr.Button(
                        "🔍 Detect Potholes", 
                        variant="primary", 
                        size="lg",
                        scale=2
                    )
                    clear_img_btn = gr.Button("🗑️ Clear", size="lg", scale=1)
                
                gr.Markdown("**Supported formats:** JPG, PNG, JPEG")
            
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Detection Statistics")
                image_info = gr.Markdown(
                    """
                    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                color: white; padding: 2rem; border-radius: 10px; text-align: center;
                                font-size: 1.5rem; font-weight: bold; min-height: 150px;
                                display: flex; align-items: center; justify-content: center;'>
                        📊 Upload an image to see detection results
                    </div>
                    """
                )
        
        gr.Markdown("---")
        gr.Markdown("### 🔍 Detection Results - Before & After Comparison")
        
        with gr.Row():
            with gr.Column(scale=1):
                image_before = gr.Image(label="📷 Original Image", height=400)
            with gr.Column(scale=1):
                image_after = gr.Image(label="✅ Detected Potholes", height=400)
        
        # Button actions
        image_button.click(
            fn=detect_potholes_image,
            inputs=image_input,
            outputs=[image_before, image_after, image_info]
        )
        
        def clear_image():
            return None, None, None, """
                    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                color: white; padding: 2rem; border-radius: 10px; text-align: center;
                                font-size: 1.5rem; font-weight: bold; min-height: 150px;
                                display: flex; align-items: center; justify-content: center;'>
                        📊 Upload an image to see detection results
                    </div>
                    """
        
        clear_img_btn.click(
            fn=clear_image,
            outputs=[image_input, image_before, image_after, image_info]
        )
    
    with gr.Tab("🎥 Video Detection"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Upload Video")
                video_input = gr.Video(
                    label="Drop your video here or click to upload",
                    height=400
                )
                with gr.Row():
                    video_button = gr.Button(
                        "🎬 Process Video", 
                        variant="primary", 
                        size="lg",
                        scale=2
                    )
                    clear_vid_btn = gr.Button("🗑️ Clear", size="lg", scale=1)
                
                gr.Markdown("**Supported format:** MP4")
                gr.Markdown("⚠️ **Note:** Video processing may take a few minutes depending on length")
            
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Processing Status")
                video_info = gr.Markdown(
                    """
                    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                color: white; padding: 2rem; border-radius: 10px; text-align: center;
                                font-size: 1.5rem; font-weight: bold; min-height: 150px;
                                display: flex; align-items: center; justify-content: center;'>
                        🎥 Upload a video to start processing
                    </div>
                    """
                )
        
        gr.Markdown("---")
        gr.Markdown("### 🎬 Processed Video Output")
        video_output = gr.Video(label="✅ Processed Video with Detections", height=500)
        
        # Button actions
        video_button.click(
            fn=detect_potholes_video,
            inputs=video_input,
            outputs=[video_output, video_info]
        )
        
        def clear_video():
            return None, None, """
                    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                color: white; padding: 2rem; border-radius: 10px; text-align: center;
                                font-size: 1.5rem; font-weight: bold; min-height: 150px;
                                display: flex; align-items: center; justify-content: center;'>
                        🎥 Upload a video to start processing
                    </div>
                    """
        
        clear_vid_btn.click(
            fn=clear_video,
            outputs=[video_input, video_output, video_info]
        )
    
    with gr.Tab("ℹ️ About"):
        gr.Markdown("""
        ## 🚀 About This Application
        
        This AI-powered pothole detection system uses **YOLOv4-Tiny**, a state-of-the-art object detection model, 
        to identify potholes in images and videos with high accuracy.
        
        ### 🎯 Features:
        - **Real-time Detection**: Fast processing of images and videos
        - **High Accuracy**: Powered by YOLOv4-Tiny deep learning model
        - **Visual Comparison**: Side-by-side before/after results
        - **Confidence Scores**: Each detection includes confidence percentage
        - **Batch Processing**: Process multiple files efficiently
        
        ### 📋 How to Use:
        1. Choose either **Image** or **Video** detection tab
        2. Upload your file (drag & drop or click to browse)
        3. Click the detect/process button
        4. View results with bounding boxes and statistics
        
        ### ⚙️ Technical Details:
        - **Model**: YOLOv4-Tiny
        - **Framework**: OpenCV DNN
        - **Input Size**: 416x416
        - **Confidence Threshold**: 60%
        - **NMS Threshold**: 40%
        
        ### 💡 Tips for Best Results:
        - Use clear, high-resolution images
        - Ensure good lighting conditions
        - Avoid blurry or heavily compressed files
        - For videos, maintain stable camera movement
        
        ---
        
        Made with ❤️ using Gradio and OpenCV
        """)
    
    # Footer
    gr.Markdown("""
    ---
    <div style='text-align: center; padding: 1rem; color: white;'>
        💡 <b>Tip:</b> For best results, use clear images/videos with good lighting conditions<br>
        🔒 Your data is processed locally and not stored on servers
    </div>
    """)

if __name__ == "__main__":
    demo.launch(share=False, favicon_path=None)
