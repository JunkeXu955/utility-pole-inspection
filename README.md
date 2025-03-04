# Utility Pole Inspection System

An intelligent web-based system for automated utility pole inspection using computer vision and AI technologies. This system combines YOLOv8-based damage detection with AI-powered report generation to streamline the inspection process.

![System Overview]([image](https://github.com/JunkeXu955/utility-pole-inspection/blob/main/img/Image.jpg))

## Features

- 🔍 Automated damage detection using YOLOv8
- 📊 Intelligent report generation using Groq AI
- 🌐 Web-based interface for easy access
- 🖼️ Advanced image processing and annotation
- 📋 Comprehensive inspection documentation

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Flask web server

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/utility-pole-inspection.git
cd utility-pole-inspection
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Project Structure

```
utility-pole-inspection/
├── app.py                # Flask web application
├── requirements.txt      # Project dependencies
├── static/              # Static files
├── templates/           # HTML templates
├── uploads/             # Upload directory
└── README.md           # This file
```

## Usage

1. Configure the application:
   - Set your Groq API key in app.py
   - Specify the path to your YOLOv8 model
   - Ensure all directories exist (static, uploads, etc.)

2. Start the Flask application:
```bash
python app.py
```

3. Access the web interface:
   - Open your browser and navigate to `http://localhost:5000`
   - Upload an image of a utility pole
   - View detection results and generated report

## Model Training

The system uses a YOLOv8 model trained on utility pole damage data. To train your own model:

1. Prepare your dataset following YOLOv8 format
2. Modify training parameters as needed
3. Follow YOLOv8 training documentation

## API Configuration

The system uses Groq API for report generation. To configure:

1. Get your API key from [Groq](https://console.groq.com)
2. Set the API key in app.py
3. Adjust report generation parameters as needed

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- YOLOv8 for object detection
- Groq for AI-powered report generation
- OpenCV for image processing

## Contact

Your Name - [@JunkeXu955](https://www.linkedin.com/in/junke-xu-45b516244/)

Project Link: [https://github.com/JunkeXu955/utility-pole-inspection](https://github.com/JunkeXu955/utility-pole-inspection) 
