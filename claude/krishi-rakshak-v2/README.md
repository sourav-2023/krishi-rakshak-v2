# Krishi Rakshak v2

AI-powered plant disease detection system. Upload a leaf image and get instant disease diagnosis with confidence scores — built for farmers and agricultural researchers.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Model | MobileNetV2 (Transfer Learning), TensorFlow/Keras |
| Backend | FastAPI, Python 3.11 |
| Frontend | React 18, Vite |
| Dataset | PlantVillage (38 classes) |

## Project Structure

```
krishi-rakshak-v2/
├── backend/
│   ├── config.py          # Constants (image size, thresholds, etc.)
│   ├── requirements.txt   # Python dependencies
│   └── .env               # Environment variables (not committed)
├── frontend/
│   ├── src/
│   └── package.json
└── model/
    ├── train.py           # MobileNetV2 training script
    ├── data/
    │   └── plantvillage/  # Dataset (not committed)
    ├── krishi_rakshak_v2.keras   # Trained model (not committed)
    ├── krishi_rakshak_v2.tflite  # TFLite export (not committed)
    └── class_names.json   # Class index map (generated after training)
```

## Setup

### Prerequisites
- Python 3.11+
- Node.js 18+
- TensorFlow 2.x compatible GPU (optional but recommended for training)

### 1. Train the Model

```bash
cd model
python train.py
```

This will:
- Train MobileNetV2 on the PlantVillage dataset (2 phases)
- Save the best model as `krishi_rakshak_v2.keras`
- Export a TFLite model as `krishi_rakshak_v2.tflite`
- Save class labels to `class_names.json`

### 2. Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Copy the environment template and configure it:
```bash
cp .env.example .env
```

Run the server:
```bash
uvicorn main:app --reload
```

API will be available at `http://localhost:8000`

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

App will be available at `http://localhost:5173`

## Model Architecture

```
MobileNetV2 (ImageNet weights, frozen)
    └── GlobalAveragePooling2D
        └── Dense(256, relu)
            └── Dropout(0.4)
                └── Dense(38, softmax)
```

**Training:**
- Phase 1 — head only, Adam lr=1e-3, 10 epochs
- Phase 2 — fine-tune last 30 layers, Adam lr=1e-5, 10 epochs
- EarlyStopping (patience=3), ModelCheckpoint on val_accuracy

## Supported Crops & Diseases

38 classes from the PlantVillage dataset covering diseases in:
Tomato, Potato, Pepper, Apple, Grape, Corn, Strawberry, Peach, Cherry, Squash, and more.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `MODEL_PATH` | Path to the trained `.keras` model file |

## License

MIT
