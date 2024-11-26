広告、チラシやLPサイトの画像からテキストを段落毎に結合するファイルです。（試作品）
座標情報も併せてappendするので、段落にハッチングしたいなどの処理にお使いいただければと思います。

This file can create Japanese paragraph with bounding_poly from LP website, ad. images by using GCP vision.


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
