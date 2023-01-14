from fastapi import FastAPI, status, Request, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from mangum import Mangum
from PIL import Image
import requests
from bs4 import BeautifulSoup


from transformers import pipeline
from transformers import CLIPProcessor, CLIPModel
zero_shot_classifier = pipeline("zero-shot-classification", model="app/bart-large-mnli-local", local_files_only=True)

fe_model = CLIPModel.from_pretrained("app/clip-vit-large-patch14-local")
fe_processor = CLIPProcessor.from_pretrained("app/clip-vit-large-patch14-local")

app = FastAPI()

@app.get('/ping', status_code=status.HTTP_200_OK)
async def ping(request: Request):
	return {
		'status': 'success',
		'made by': 'hafiz <3'
	}

@app.get('/evaluate-job-risk', status_code=status.HTTP_200_OK)
async def evaluate_job_risk(job_title: str, request: Request):
	labels = ['danger', 'office']
	
	job_title = job_title.replace(' ', '+')
	url = f'https://www.google.com/search?q={job_title}+job+description'
	page = requests.get(url)
	soup = BeautifulSoup(page.text, 'html.parser')
	job_description = soup.select('.BNeawe')[0].text

	result = zero_shot_classifier(job_description, labels)
	danger_index = result['labels'].index('danger')
	risk_probability = result['scores'][danger_index]
	return {
		'job_title': job_title.replace('+', ' '),
		'job_description': job_description,
		'risk_probability': risk_probability
	}

# @app.get('/extract-topics', status_code=status.HTTP_200_OK)
# async def extract_topics(job_description: str, request: Request):
# 	labels = ['disease', 'construction', 'widow', 'old']
# 	result = zero_shot_classifier(job_description, labels)
# 	return {
# 		'label': result['labels'][0],
# 		'probability': result['scores'][0]
# 	}

@app.post('/fr-identify-obesity', status_code=status.HTTP_200_OK)
async def fr_identify_obesity(file: UploadFile = File()):
	image = Image.open(file.file)

	inputs = fe_processor(text=['obesity', 'healthy'], images=image, return_tensors="pt", padding=True)
	outputs = fe_model(**inputs)
	logits_per_image = outputs.logits_per_image # this is the image-text similarity score
	probs = logits_per_image.softmax(dim=1) 

	return {
		'obese_probability': probs[0][0].item(),
	}

@app.get('/risk-score', status_code=status.HTTP_200_OK)
async def risk_score(age: int, is_smoker: bool, job_risk_proba: bool, has_existing_health_condition: bool, 
	is_family_has_existing_health_condition: bool, request: Request):

	risk_score = 0
	if age >= 60:
		risk_score += 2
	elif age >= 40:
		risk_score += 1

	if is_smoker:
		risk_score += 2
	
	if job_risk_proba >= 0.7:
		risk_score += 2
	elif job_risk_proba >= 0.4:
		risk_score += 1

	if has_existing_health_condition:
		risk_score += 2

	if is_family_has_existing_health_condition:
		risk_score += 1

	risk_tier = 1
	if risk_score > 6:
		risk_tier = 3
	elif risk_score > 3:
		risk_tier = 2

	return {
		'risk_tier': risk_tier,
		'risk_score': risk_score
	}



origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
	expose_headers=["*"],
)

handler = Mangum(app=app)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)