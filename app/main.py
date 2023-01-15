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
		'obesity_probability': probs[0][0].item(),
	}

def evaluate_job_risk(job_title):
	labels = ['danger', 'office']
	job_title = job_title.replace(' ', '+')
	url = f'https://www.google.com/search?q={job_title}+job+description'
	page = requests.get(url)
	soup = BeautifulSoup(page.text, 'html.parser')
	job_description = soup.select('.BNeawe')[0].text

	result = zero_shot_classifier(job_description, labels)
	danger_index = result['labels'].index('danger')
	risk_probability = result['scores'][danger_index]

	return risk_probability

@app.get('/risk-score', status_code=status.HTTP_200_OK)
async def risk_score(
		age: int,
		job_title: str,
		existing_condition: str,
		family_history: str,
		smoker: bool,
		married: bool,
		obesity_probability: float
	):

	risk_score = 0
	if age >= 60:
		risk_score += 2
	elif age >= 40:
		risk_score += 1

	job_risk_probability = evaluate_job_risk(job_title)
	if job_risk_probability > 0.7:
		risk_score += 2
	elif job_risk_probability > 0.5: 
		risk_score += 1
		
	if existing_condition != '-':
		risk_score += 2
	
	if family_history != '-':
		risk_score += 1

	if smoker:
		risk_score += 2
	
	if obesity_probability > 0.8:
		risk_score += 2
	elif obesity_probability > 0.5:
		risk_score += 1

	risk_tier = 1
	if risk_score >= 7:
		risk_tier = 3
	elif risk_score >= 4:
		risk_tier = 2

	return {
		'risk_tier': risk_tier,
		'risk_score': risk_score
	}


@app.get('/extract-keywords', status_code=status.HTTP_200_OK)
async def extract_keywords(
		age: int,
		job_title: str,
		existing_condition: str,
		family_history: str,
		smoker: bool,
		married: bool,
		obesity_probability: float
	):
	keywords = []

	if age > 50:
		keywords.append('old')

	labels = ['construction', 'office']
	job_title = job_title.replace(' ', '+')
	url = f'https://www.google.com/search?q={job_title}+job+description'
	page = requests.get(url)
	soup = BeautifulSoup(page.text, 'html.parser')
	job_description = soup.select('.BNeawe')[0].text

	result = zero_shot_classifier(job_description, labels)
	danger_index = result['labels'].index('danger')
	risk_probability = result['scores'][danger_index]

	if risk_probability > 0.5:
		keywords.append('construction')

	if smoker:
		keywords.append('smoking')

	if married:
		keywords.append('married')

	if obesity_probability > 0.5:
		keywords.append('obesity')
		keywords.append('disease')

	if existing_condition or family_history:
		keywords.append('disease')

	return keywords

@app.get('/evaluate-assessment', status_code=status.HTTP_200_OK)
async def evaluate_assessment(
		age: int,
		job_title: str,
		gender: str,
		existing_condition: str,
		family_history: str,
		smoker: bool,
		married: bool,
		dp_url: str
	):

	obesity_probability = fr_check_obesity(dp_url)
	print(obesity_probability)
	risk_score, risk_tier = calculate_risk(
		age,
		job_title,
		gender,
		existing_condition,
		family_history,
		smoker,
		married,
		obesity_probability
	)
	keywords = extract_keywords(
		age,
		job_title,
		gender,
		existing_condition,
		family_history,
		smoker,
		married,
		obesity_probability
	)

	
	return {
		'risk_tier': risk_tier,
		'risk_score': risk_score,
		'keywords': keywords
	}

def fr_check_obesity(dp_url):
	image = Image.open(requests.get(dp_url, stream=True).raw)
	inputs = fe_processor(text=['obesity', 'healthy'], images=image, return_tensors="pt", padding=True)
	outputs = fe_model(**inputs)
	logits_per_image = outputs.logits_per_image # this is the image-text similarity score
	obesity_probability = logits_per_image.softmax(dim=1)[0][0].item()

	return obesity_probability

def calculate_risk(
		age,
		job_title,
		gender,
		existing_condition,
		family_history,
		smoker,
		married,
		obesity_probability
	):

	risk_score = 0
	if age >= 60:
		risk_score += 2
	elif age >= 40:
		risk_score += 1

	job_risk_probability = evaluate_job_risk(job_title)
	if job_risk_probability > 0.7:
		risk_score += 2
	elif job_risk_probability > 0.5: 
		risk_score += 1
		
	if existing_condition != '-':
		risk_score += 2
	
	if family_history != '-':
		risk_score += 1

	if smoker:
		risk_score += 2
	
	if obesity_probability > 0.8:
		risk_score += 2
	elif obesity_probability > 0.5:
		risk_score += 1

	risk_tier = 1
	if risk_score >= 7:
		risk_tier = 3
	elif risk_score >= 4:
		risk_tier = 2

	return risk_score, risk_tier

def extract_keywords(
		age,
		job_title,
		gender,
		existing_condition,
		family_history,
		smoker,
		married,
		obesity_probability
	):

	keywords = []
	if age > 50:
		keywords.append('old')

	labels = ['construction', 'office']
	job_title = job_title.replace(' ', '+')
	url = f'https://www.google.com/search?q={job_title}+job+description'
	page = requests.get(url)
	soup = BeautifulSoup(page.text, 'html.parser')
	job_description = soup.select('.BNeawe')[0].text

	result = zero_shot_classifier(job_description, labels)
	construction_index = result['labels'].index('construction')
	risk_probability = result['scores'][construction_index]

	if risk_probability > 0.5:
		keywords.append('construction')

	if smoker:
		keywords.append('smoking')

	if married:
		keywords.append('married')

	if gender == 'Female':
		keywords.append('women')

	if obesity_probability > 0.5:
		keywords.append('obesity')
		keywords.append('disease')

	if existing_condition or family_history:
		keywords.append('disease')

	# remove duplicates
	keywords = list(dict.fromkeys(keywords))

	return keywords



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