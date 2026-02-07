# Plant Disease Classifier

A full-stack machine learning web application designed for automated plant health diagnosis. The system allows users to upload images of plant leaves, processes them using a Deep Learning model, and provides a real-time classification with confidence scores while maintaining a persistent diagnostic history.

## Tech Stack

* **Deep Learning Framework**: PyTorch
* **Model Architecture**: ResNet18 (Transfer Learning)
* **Backend**: FastAPI
* **Server**: Uvicorn
* **Database**: SQLite with SQLAlchemy ORM
* **Frontend**: HTML5, JavaScript (Fetch API), Tailwind CSS
* **Image Handling**: Pillow (PIL)

## Key Features

* **Instant Inference**: High-speed image classification using a pre-trained ResNet18 model.
* **Probability Distribution**: Displays secondary predictions to provide transparency in cases of low model confidence.
* **Dynamic Image Preview**: Immediate local rendering of the uploaded image before server processing.
* **Diagnostic History**: Persistent storage of analysis results (date, diagnosis, and confidence) in a local database.
* **Modern UI**: Responsive and clean interface built with utility-first CSS.

## Project Structure

```text
ML_PROJECT_PLANTS/
├── Backend/
│   ├── main.py            # FastAPI application logic and ML inference
│   ├── database.py        # SQLAlchemy engine and session configuration
│   ├── models.py          # Database schema definitions
│   ├── crud.py            # Create/Read operations for the database
│   └── uploads/           # Storage for uploaded diagnostic images
├── Model/
│   ├── my_plant_model.pth # Trained PyTorch model weights
│   └── classes.txt        # Label mapping for 15 plant disease classes
└── Frontend/
    └── index.html         # Single Page Application (SPA) frontend

```

## Installation and Setup

### 1. Prerequisites

Ensure you have Python 3.8+ installed on your system.

### 2. Environment Setup

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/yourusername/plant-disease-classifier.git
cd plant-disease-classifier
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

```

### 3. Install Dependencies

```bash
pip install torch torchvision fastapi uvicorn sqlalchemy pillow python-multipart

```

### 4. Running the Application

Launch the server using the module flag to ensure correct relative imports:

```bash
python3 -m Backend.main

```

The application will be available at `http://localhost:8000`.

## Database Configuration

The system utilizes **SQLite** in **Write-Ahead Logging (WAL)** mode. This configuration enables the FastAPI backend to perform concurrent write operations while allowing external tools (such as DB Browser for SQLite or VS Code extensions) to read the data in real-time without locking the database file.

## License

This project was developed for educational purposes as a demonstration of integrating Deep Learning models into a production-ready web infrastructure.

---
