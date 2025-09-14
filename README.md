```markdown
# 🍽 NutriMenu AI - Smart Menu Generator

##  Project Overview

NutriMenu AI is an intelligent menu generation system that combines computer vision, natural language processing, and nutrition intelligence to create personalized meal plans. The system uses your trained Indian food classification model to identify dishes from images and generates nutritionally balanced menus based on user preferences.

## 🏗 Project Structure

```
nutrimenu-ai/
│
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI application
│   │   ├── models.py           # Pydantic models
│   │   ├── nutrition_engine.py # Nutrition calculation engine
│   │   ├── menu_generator.py   # Menu generation algorithms
│   │   └── nlp_processor.py    # NLP processing for text inputs
│   ├── ml_models/
│   │   ├── final_restarted_model.keras  # Your trained model
│   │   └── label_encoder.pkl   # Label encoder
│   ├── data/
│   │   └── indian_food_nutrition.csv  # Nutrition database
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ImageUpload.js
│   │   │   ├── NutritionDisplay.js
│   │   │   ├── MenuGenerator.js
│   │   │   └── VoiceInput.js
│   │   ├── App.js
│   │   ├── App.css
│   │   └── index.js
│   ├── public/
│   ├── package.json
│   └── Dockerfile
│
├── notebooks/
│   ├── model_training.ipynb
│   └── nutrition_analysis.ipynb
│
├── scripts/
│   ├── setup_database.py
│   └── process_nutrition_data.py
│
├── .env
├── docker-compose.yml
├── README.md
└── requirements.txt
```
