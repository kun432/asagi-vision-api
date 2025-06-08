# Asagi-Vision-API

これは、Asagi-VisionモデルのためのFastAPIベースのAPIサーバーです。

## 前提条件

- Python 3.12 以降
- `uv`

## セットアップ

1.  **リポジトリのクローン:**
    ```bash
    git clone <リポジトリURL>
    cd asagi-vision-api
    ```

2.  **依存関係のインストール:**
    ```bash
    uv sync
    ```

3.  **環境変数:**
    `.env.example` を参考に `.env` ファイルを作成し、必要に応じて以下の環境変数を設定します。

    -   `HOST`: APIサーバーがリッスンするホストアドレスです。デフォルトは `0.0.0.0` です。
    -   `PORT`: APIサーバーがリッスンするポート番号です。デフォルトは `8000` です。
    -   `LOG_LEVEL`: アプリケーションのログレベルを設定します (例: `info`, `debug`)。デフォルトは `info` です。
    -   `QUANTIZATION_MODE`: モデル読み込み時の量子化モードを指定します（bitsandbytes）。注意: 量子化すると精度がかなり落ちる印象があります。
        -   `4bit`: 4bit量子化を有効にします。
        -   `8bit`: 8bit量子化を有効にします。
        -   `none` (または未設定): 量子化を行いません (デフォルト)。
    -   `MODEL_NAME`: 使用するHugging Faceのモデル名を指定します。デフォルトは `MIL-UT/Asagi-2B` です。
        -   `MIL-UT/Asagi-2B` （デフォルト）
        -   `MIL-UT/Asagi-4B`
        -   `MIL-UT/Asagi-8B`
        -   `MIL-UT/Asagi-14B`

    `.env.example` には各変数の設定例が記載されています。通常、開発環境では `.env.example` をコピーして `.env` を作成し、値を調整します。
    ```bash
    cp .env.example .env
    ```
    その後、`.env` ファイルを編集してください。

## サーバーの実行

```bash
uv run server.py
```
サーバーは通常 `http://localhost:8000` で起動します。

## curl を使用したテスト

`/generate` エンドポイントは `curl` を使用してテストできます。

**1. 画像の準備:**
   画像ファイル (例: `assets/kobe.jpg`) を用意してください。

**2. Base64エンコードされた画像でのテスト:**

   a. **画像をBase64にエンコード:**
      ターミナルを開き、次のコマンドを実行します（画像は適宜変更してください）

      Linuxの場合
      ```bash
      IMAGE_B64=$(base64 -w 0 assets/kobe.jpg)
      ```

      Macの場合
      ```bash
      IMAGE_B64=$(base64 -w 0 -i assets/kobe.jpg)
      ```
      このコマンドは画像をエンコードし、Base64文字列をシェル変数 `IMAGE_B64` に保存します。`-w 0` オプションは、出力が1行になるようにします。

   b. **`payload.json` ファイルの作成:**
      次の内容で `payload.json` という名前のファイルを作成します。`image` フィールドはBase64文字列を使用します。
      ```bash
      cat <<EOF > payload.json
      {
        "prompt": "この画像について説明してください。",
        "image": "$IMAGE_B64",
        "max_new_tokens": 128,
        "temperature": 0.1
      }
      EOF
      ```

   c. **`curl` を使用してリクエストを送信:**
      FastAPIサーバーが実行中であることを確認してから、以下を実行します:
      ```bash
      curl -X POST "http://localhost:8000/generate" \
      -H "Content-Type: application/json" \
      -d @payload.json
      ```

**3. 画像URLでのテスト:**

   a. **`payload.json` ファイルの作成:**
      次の内容で `payload.json` という名前のファイルを作成します。画像URLは適宜変更してください。
      ```bash
      cat <<EOF > payload.json
      {
        "prompt": "この画像について説明してください。",
        "image": "https://storage.googleapis.com/zenn-user-upload/82968d23b6c5-20250228.jpg",
        "max_new_tokens": 128,
        "temperature": 0.1
      }
      EOF
      ```

   b. **`curl` を使用してリクエストを送信:**
      FastAPIサーバーが実行中であることを確認してから、以下を実行します:
      ```bash
      curl -X POST "http://localhost:8000/generate" \
      -H "Content-Type: application/json" \
      -d @payload.json
      ```

**期待されるレスポンス:**
モデルによって生成されたテキストを含むJSONレスポンス (`generated_text` フィールド) を受け取るはずです。

## エンドポイント

-   **`/generate`**: (POST) プロンプトとオプションの画像 (URLまたはBase64文字列として提供) に基づいてテキストを生成します。
-   **`/docs`**: FastAPIによって生成されたAPIドキュメント (Swagger UI)。
-   **`/redoc`**: FastAPIによって生成されたAPIドキュメント (ReDoc)。
