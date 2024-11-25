import os
import requests  # HTTPリクエストを送信するためのモジュール
from fluent_japanese_vision import FluentJapaneseVision
import base64

# Google Cloud Vision APIキー

API_KEY = os.getenv("GCP_KEY")
API_URL = f"https://vision.googleapis.com/v1/images:annotate?key={API_KEY}"

# 画像データからテキストと座標を抽出する関数
def img_to_text_with_position(image_content: bytes) -> list[dict]:
    """
    スクショをVision APIに送り、text_sectionをマージして返す。
    Args:
        image_content(bytes): 画像
    Returns:
        list[dict]: fluent_japanese_visionでマージしたtext_sectionを全て格納したリスト
    """
    # 画像データをBase64エンコード
    encoded_image: str = base64.b64encode(image_content).decode("utf-8")

    # Vision APIリクエストデータの準備
    request_data: dict = {
        "requests": [
            {
                "image": {"content": encoded_image},
                "features": [{"type": "TEXT_DETECTION"}],
            }
        ]
    }

    # Vision APIにリクエスト送信
    vision_api_response: requests.Response = requests.post(API_URL, json=request_data)
    vision_api_response.raise_for_status()  # エラーチェック
    json_response: dict = vision_api_response.json()

    # 近いtext_sectionを結合する。
    fjv: FluentJapaneseVision = FluentJapaneseVision()
    combined_sections_list: list[dict] = fjv.run(json_response)

    return combined_sections_list


def document_analyzer() -> dict:
    """
    画像をGoogle Vision APIに送って文字を読む。
    その後必要に応じてtext_sectionを近隣同士で結合して、
    新しいtext_sectionそれぞれに対してテキストと座標とフォントサイズを書き出した辞書を返す。
    """
    image_content = "<images.jpeg>"
    with open("images.jpeg", "rb") as image_file:
        image_content = image_file.read()
    # 画像からテキストと座標を抽出
    extracted_sections: list[dict] = img_to_text_with_position(image_content)
    print(extracted_sections)
    
if __name__ == "__main__":
    document_analyzer()
