from flask import Flask, request, render_template
import base64
import os
from werkzeug.utils import secure_filename
from groq import Groq
import logging
from ultralytics import YOLO
import cv2
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

# Initialize Groq client (replace with your actual API key)
client = Groq(api_key="")  # Replace with your actual API key

# Load YOLOv8 model (replace with your trained model path)
model = YOLO('C:/Users/junke/PycharmProjects/pythonProject1/runs/detect/yolov8_damage_detection3/weights/best.pt')

# Function to read image and convert it to base64
def read_image_as_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logging.error(f"Error reading image: {e}")
        return None

def detect_damages(image_path):
    """Detect damages using the YOLOv8 model and annotate the image with bounding boxes and arrows."""
    # Perform prediction
    results = model.predict(source=image_path, save=False, imgsz=640)

    # Read the original image
    image = cv2.imread(image_path)

    # Define padding size for each side
    top_padding = 100
    bottom_padding = 100
    left_padding = 200
    right_padding = 200

    # Add padding (border) around the image
    image = cv2.copyMakeBorder(
        image,
        top=top_padding,
        bottom=bottom_padding,
        left=left_padding,
        right=right_padding,
        borderType=cv2.BORDER_CONSTANT,
        value=[255, 255, 255]  # White border
    )

    damages = []
    label_positions = []  # Keep track of label positions to prevent overlap
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls)
            # Retrieve the class name
            class_name = result.names[cls_id]

            confidence = box.conf  # Confidence score
            xyxy = box.xyxy[0]  # Bounding box coordinates

            # Convert coordinates to integers and adjust based on padding
            x1, y1, x2, y2 = map(int, xyxy.tolist())
            x1 += left_padding
            y1 += top_padding
            x2 += left_padding
            y2 += top_padding

            damage_info = {
                "class": class_name,
                "confidence": confidence,
                "coordinates": [x1, y1, x2, y2]
            }
            damages.append(damage_info)

            # Draw the bounding box
            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                color=(0, 255, 0),  # Green color (BGR)
                thickness=2
            )

            # Calculate the center of the bounding box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Determine the closest edge (including the padding)
            img_height, img_width = image.shape[:2]
            distances = {
                'top': center_y,
                'bottom': img_height - center_y,
                'left': center_x,
                'right': img_width - center_x
            }
            closest_edge = min(distances, key=distances.get)

            # Set arrow endpoint based on the closest edge
            if closest_edge == 'top':
                arrow_end_x = center_x
                arrow_end_y = 0
            elif closest_edge == 'bottom':
                arrow_end_x = center_x
                arrow_end_y = img_height - 1
            elif closest_edge == 'left':
                arrow_end_x = 0
                arrow_end_y = center_y
            else:  # 'right'
                arrow_end_x = img_width - 1
                arrow_end_y = center_y

            # Draw the arrow
            cv2.arrowedLine(
                image,
                (center_x, center_y),
                (arrow_end_x, arrow_end_y),
                color=(0, 0, 255),  # Red color (BGR)
                thickness=2,
                tipLength=0.05
            )

            # Increase font size and thickness for the label
            label = f"{class_name}"
            font_scale = 1.0  # Font size
            thickness = 2     # Font thickness

            (label_width, label_height), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                thickness
            )

            # Adjust label position based on arrow direction
            label_offset = 40  # Increase this value to prevent overlap
            if closest_edge == 'top':
                # Position the label below the arrow end point
                label_x = arrow_end_x - label_width // 2
                label_y = arrow_end_y + label_height + label_offset
            elif closest_edge == 'bottom':
                # Position the label above the arrow end point
                label_x = arrow_end_x - label_width // 2
                label_y = arrow_end_y - label_offset
            elif closest_edge == 'left':
                # Position the label to the right of the arrow end point
                label_x = arrow_end_x + label_offset
                label_y = arrow_end_y + label_height // 2
            elif closest_edge == 'right':
                # Position the label to the left of the arrow end point
                label_x = arrow_end_x - label_width - label_offset
                label_y = arrow_end_y + label_height // 2

            # Ensure the label is within image boundaries
            label_x = max(0, min(label_x, img_width - label_width))
            label_y = max(label_height, min(label_y, img_height - baseline))

            # Adjust label position if overlapping with previous labels
            for prev_x, prev_y, prev_w, prev_h in label_positions:
                if abs(label_x - prev_x) < (label_width + prev_w) // 2 and abs(label_y - prev_y) < (label_height + prev_h) // 2:
                    label_y += label_height + 10  # Move label down to prevent overlap

            # Save current label position
            label_positions.append((label_x, label_y, label_width, label_height))

            # Draw the label text horizontally
            cv2.putText(
                image,
                label,
                (int(label_x), int(label_y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA
            )

    # Save the annotated image
    dest_dir = os.path.join('static', 'uploads')
    os.makedirs(dest_dir, exist_ok=True)
    annotated_image_name = os.path.splitext(os.path.basename(image_path))[0] + "_annotated.jpg"
    annotated_image_path = os.path.join(dest_dir, annotated_image_name)

    # Write the annotated image
    cv2.imwrite(annotated_image_path, image)
    logging.info(f"Annotated image saved to: {annotated_image_path}")

    return damages, annotated_image_path

def generate_report_with_groq(base64_image, additional_info, damage_summary):
    if base64_image is None:
        return "Error: Image could not be processed."

    cot_instructions = f"""
    You are an expert utility pole inspector with advanced knowledge of material science, structural integrity, and environmental factors.

    **Detected Damages:**
    {damage_summary}

    **Additional Information:**
    {additional_info}

    **Instructions:**

    Analyze the provided image of a utility pole along with the detected damages and additional information above. Follow the steps carefully and provide detailed explanations in each step.

    **Steps:**

    1. **Material Identification**: Determine the material of the pole (wood, metal, concrete, or composite). Explain the characteristics (texture, color, construction details) that led to your conclusion.

    2. **Defect Detection**: Based on the detected damages, provide a detailed analysis of each identified issue, including its potential causes and implications.

    3. **Environmental Impact**: Assess how environmental factors and the detected damages may be affecting the pole.

    4. **Structural Integrity**: Evaluate the overall stability of the pole, considering the detected damages.

    5. **Safety Compliance**: Check for compliance with safety standards, taking into account the detected damages.

    6. **Maintenance Recommendations**: Offer detailed suggestions for maintenance or repairs specific to the detected damages.

    7. **Conclusion**: Summarize your findings and provide an overall assessment.

    **Provide your analysis in the following report format:**

    ---

    **Utility Pole Inspection Report**

    - **Material Identification**:
      - *[Your detailed analysis]*

    - **Defect Detection**:
      - *[Your detailed analysis]*

    - **Environmental Impact**:
      - *[Your detailed analysis]*

    - **Structural Integrity**:
      - *[Your detailed analysis]*

    - **Safety Compliance**:
      - *[Your detailed analysis]*

    - **Maintenance Recommendations**:
      - *[Your detailed suggestions]*

    - **Conclusion**:
      - *[Your overall assessment]*

    ---

    **Note**: Make sure to integrate the detected damages into your analysis.
    """

    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": cot_instructions},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            model="llama-3.2-90b-vision-preview"
        )

        ai_report = response.choices[0].message.content
        return ai_report
    except Exception as e:
        logging.error(f"Error generating report: {e}")
        return "Error: Report generation failed."

def analyze_report(report):
    """Analyze the generated report, highlighting accurate parts in green and inaccurate/unhelpful parts in red."""
    analysis_prompt = f"""
    As an expert utility pole inspector, please analyze the following report.

    - Surround **accurate** and **helpful** statements with `[GREEN]` at the beginning and `[/GREEN]` at the end.
    - Surround **inaccurate** or **unhelpful** statements with `[RED]` at the beginning and `[/RED]` at the end.
    - Leave neutral statements without tags.

    Report:
    {report}

    Return the annotated report with the highlights.
    """

    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": analysis_prompt
                }
            ],
            model="llama-3.2-90b-vision-preview"
        )

        analyzed_report = response.choices[0].message.content

        # Log the raw analyzed report
        logging.info(f"Raw analyzed report:\n{analyzed_report}")

        # Replace custom tags with HTML <span> elements for color
        analyzed_report = analyzed_report.replace('[GREEN]', '<span style="background-color: green;">')
        analyzed_report = analyzed_report.replace('[/GREEN]', '</span>')
        analyzed_report = analyzed_report.replace('[RED]', '<span style="background-color: red;">')
        analyzed_report = analyzed_report.replace('[/RED]', '</span>')

        # Return analyzed report directly without sanitization
        return analyzed_report

    except Exception as e:
        logging.error(f"Error analyzing report: {e}")
        return report  # Return the original report if analysis fails

def generate_report_for_images(image_filenames, additional_info):
    """Generate reports for multiple images by processing each image in parallel."""
    reports = []
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(generate_report_for_image, filename, additional_info)
            for filename in image_filenames
        ]
        for future in futures:
            reports.append(future.result())
    return reports

def generate_report_for_image(filename, additional_info):
    """Generate a detailed report for a single image by detecting damages and analyzing the report."""
    image_path = os.path.join("static", "uploads", filename)
    base64_image = read_image_as_base64(image_path)

    # Detect damages and get the annotated image
    damages, annotated_image_path = detect_damages(image_path)

    # Create a damage summary
    damage_summary = "\n".join([
        f"- Detected **{d['class']}** with **{d['confidence'].item()*100:.2f}%** confidence at coordinates {d['coordinates']}"
        for d in damages
    ])

    # Generate the initial AI report
    ai_report = generate_report_with_groq(base64_image, additional_info, damage_summary)

    # Analyze the report, marking incorrect or unhelpful sections
    analyzed_report = analyze_report(ai_report)

    # Prepare the annotated image path for the template
    if annotated_image_path:
        annotated_image_relative_path = os.path.relpath(annotated_image_path, 'static').replace('\\', '/')
    else:
        annotated_image_relative_path = None

    return {
        'image_filename': filename,
        'annotated_image': annotated_image_relative_path,
        'ai_report': analyzed_report  # Use the analyzed report with highlights
    }

@app.route('/')
def home():
    """Render the homepage with the upload form."""
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_images():
    """Handle the image upload and report generation."""
    if 'images' not in request.files:
        return "No images uploaded.", 400

    image_files = request.files.getlist('images')
    image_filenames = []

    # Save uploaded images
    os.makedirs('static/uploads', exist_ok=True)
    for image in image_files:
        filename = secure_filename(image.filename)
        image.save(os.path.join('static', 'uploads', filename))
        image_filenames.append(filename)

    # Get additional text input
    additional_info = request.form.get('additional_info', '')

    # Generate reports
    reports = generate_report_for_images(image_filenames, additional_info)

    # Render the report page
    return render_template('report.html', reports=reports)

if __name__ == '__main__':
    app.run(debug=True)
