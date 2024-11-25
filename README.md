This file can create Japanese paragraph from LP website images by using GCP vision.

Setup
1. GET Google Vision API

2.set your API key(add $env or rewriting below code.)

```
export GCP_KEY=<your_API_KEY>
```

or 

```
API_KEY = os.getenv("<YOUR_GCP_API_ENV_NAME>")
```
3.Install requirements.txt on your venv

```
python3.10 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

R
RUN
```
python main.py
```
