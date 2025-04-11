# 🩺 VadiyaVarta — AI-Powered Disease Prediction & Doctor Recommendation System

**VadiyaVarta** is an intelligent healthcare assistant that utilizes **NLP, BioBERT, and machine learning** to accurately predict diseases from user-reported symptoms and intelligently match them with the most suitable doctors. It simplifies access to healthcare by offering a unified platform with features like symptom-based diagnosis, skin disease identification, doctor recommendation, and appointment scheduling.

---

## 🚀 Features

- 🔬 **AI-Based Symptom to Disease Prediction** (using BioBERT)
- 🧠 **Skin Disease Classification** (using custom-trained CNN on ISIC dataset)
- 👩‍⚕️ **Doctor Recommendation System** based on predicted disease & specialization
- 📅 **Appointment Booking & Management**
- 📊 **Role-Based Dashboards** (Patient & Doctor view)
- 🌐 **FastAPI Backend with Frontend Integration**
- 📄 **Dynamic Reports & Visual Feedback**
- ✅ **OTP-Based Transaction Verification (in fraud protection module)**

---

## 🛠️ Tech Stack

| Layer        | Technology                     |
|--------------|--------------------------------|
| Backend      | FastAPI, Python                |
| Frontend     | HTML/CSS, JavaScript (basic UI)|
| ML/NLP Models| BioBERT (Hugging Face), CNN    |
| Libraries    | Transformers, Pandas, Scikit-learn, Keras, OpenCV |
| Dataset(s)   | ISIC 2019 (Skin Images), Custom CSVs |
| Database     | SQLite (via SQLAlchemy)        |
| Deployment   | Streamlit for prototyping, GitHub for version control |
| Others       | Google Colab (Model Training), Jupyter Notebooks, HuggingFace |

---

## 📂 Project Structure

```
VadiyaVarta/
├── models/                    # Pretrained models and weights
├── bio_bert/                 # BioBERT disease prediction model
├── skin_classifier/          # CNN for skin disease classification
├── frontend/                 # Frontend templates and static files
├── backend/                  # FastAPI backend with APIs
├── datasets/                 # Labeled CSVs and raw datasets
├── evaluation_results/       # Model outputs & scoring results
├── report_generation/        # DOCX / PDF report generation scripts
└── main.py                   # Entrypoint for running the application
```

---

## 🧠 AI Models

### 1. Disease Prediction (BioBERT)
- Fine-tuned on biomedical symptom-disease data.
- Takes free-form symptom input and returns top probable diseases.

### 2. Skin Disease Classification
- CNN trained on ISIC 2019 dataset with 9 output classes.
- Preprocessing includes resizing, augmentation, and normalization.

---

## ⚙️ How to Run Locally

1. **Clone the repo**
   ```bash
   git clone https://github.com/HetPatel1978/VadiyaVarta.git
   cd VadiyaVarta
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run FastAPI app**
   ```bash
   uvicorn backend.main:app --reload
   ```

5. **Launch Streamlit UI (optional prototype)**
   ```bash
   streamlit run frontend/app.py
   ```

---

## 📊 Example Use Case

1. User types: _"I've had a headache, nausea, and light sensitivity for 2 days"_
2. BioBERT processes this input → Predicts: `Migraine`
3. System fetches top-rated neurologists from the doctor database
4. User books appointment or downloads a report

---

## 📎 Resources & Datasets

- [BioBERT Model - Hugging Face](https://huggingface.co/dmis-lab/biobert-base-cased-v1.1)
- [ISIC Skin Dataset](https://challenge2019.isic-archive.com/)
- Custom Labeled CSVs for Doctor Specialization Mapping

---

## 📌 Future Improvements

- 🔁 Retraining with real-time user data
- 🧾 Health report summarization via OCR integration
- 🧬 Adding drug recommendation based on diagnosis
- 🏥 Integration with real hospital APIs

---

## 👨‍💻 Contributors

- **Het Patel** — Developer & AI Engineer  
- **Ansh**, **Surya** — ML Module Integration  
- Special thanks to **ISIC** and **Hugging Face** community

---

## 📜 License

This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for details.

---

## 🌐 Demo / Preview

A working demo will be uploaded soon!

---