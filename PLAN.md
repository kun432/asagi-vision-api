# 開発計画: Sarashina2-Vision OpenAI互換API

このドキュメントは、FastAPIを使用して `sbintuitions/sarashina2-vision-8b` モデルのためのOpenAI互換APIサーバーを開発する計画の概要を示します。

## フェーズ1: 基本的なFastAPIサーバーのセットアップ (OpenAI互換 - コアエンドポイント)

1.  **プロジェクト構成と初期設定**:
    *   依存関係（例: `python-dotenv`）について `pyproject.toml` を確認・更新します。
    *   メインのFastAPIアプリケーションスクリプトとして `server.py` を作成します。
    *   環境変数管理のために `.env` を、リポジトリ用に `.env.example` をセットアップします。（APIキー以外の設定に利用可能性あり）
    *   `.gitignore` が適切に設定されていることを確認します。
    *   プロジェクト概要とセットアップ手順を記載した `README.md` を更新または作成します。
2.  **認証**:
    *   このAPIサーバー自体では認証処理を行いません。
    *   認証は、APIゲートウェイやリバースプロキシなど、上位のレイヤーで実施されることを想定します。
3.  **`/v1/models` エンドポイントの実装**:
    *   `docs/requirements_and_specs.md` に基づいて `GET /v1/models` エンドポイントを作成します。
    *   OpenAI APIスキーマに準拠し、`sbintuitions/sarashina2-vision-8b` モデルのみを含むリストを返します。
    *   レスポンス構造のためにPydanticモデルを定義します。
4.  **`/v1/chat/completions` エンドポイント - スキーマと基本構造**:
    *   OpenAIの仕様および `docs/requirements_and_specs.md` に基づいて、`ChatCompletionRequest`、`ChatCompletionResponse`、`ChatCompletionChunk` のためのPydanticモデルを定義します。
    *   `POST /v1/chat/completions` エンドポイントの基本構造を作成します。
    *   初期段階では、固定レスポンスを返すか、完全なモデル推論なしでストリーミングの準備（`EventSourceResponse`など）を行います。
    *   `stream: true` パラメータを処理できるように設計します。

## フェーズ2: Sarashina2-Visionモデルの統合

5.  **モデルの読み込みと初期化**:
    *   Hugging Face `transformers` を使用して `sbintuitions/sarashina2-vision-8b` モデルとプロセッサをロードするロジックを実装します。
    *   `docs/sarashina2-vision-8b.md` で指定されているように `trust_remote_code=True` を使用します。
    *   アプリケーション起動時にモデルをロードし、利用可能であればGPUに配置します（GPUは要件です）。
6.  **画像入力処理**:
    *   `ChatCompletionRequest` の `messages` 配列内で渡される画像入力（Base64エンコードされたデータURL）を処理するロジックを実装します。
    *   画像をデコードし、モデルが受け入れ可能な形式に変換します（例: PILを使用）。
7.  **`/v1/chat/completions` の推論ロジック**:
    *   実際のモデル推論ロジックを `/v1/chat/completions` エンドポイントに統合します。
    *   入力メッセージ（テキストと画像）からモデルへのプロンプトを構築します。
    *   `max_tokens`、`temperature` などのパラメータを考慮して、推論のために `model.generate()` を利用します。
    *   モデルの出力をOpenAI互換のレスポンス構造にフォーマットします。
    *   `stream: true` の場合にストリーミングレスポンスを完全に実装します。

## フェーズ3: コンテナ化とCI/CD (GitHub Actions)

8.  **Dockerfileの作成**:
    *   アプリケーションをコンテナ化するための `Dockerfile` を作成します。
    *   適切なベースイメージ（例: Python公式、NVIDIA CUDA）を選択します。
    *   依存関係のインストール（`uv pip install`）、ソースコードのコピー、サーバー起動コマンドの設定（例: Uvicorn経由の`CMD`または`ENTRYPOINT`）のステップを定義します。
    *   モデルファイルの取り扱いに対処します（`docs/requirements_and_specs.md` の推奨はボリュームマウントです）。
9.  **CI/CDのためのGitHub Actionsワークフロー**:
    *   ワークフローファイル（例: `.github/workflows/docker-publish.yml`）を作成します。
    *   特定ブランチ（例: `main`）へのプッシュをトリガーとするようにワークフローを設定します。
    *   Dockerイメージをビルドし、GitHub Container Registry (GHCR) にプッシュするジョブを定義します。
    *   GHCRへのログイン、イメージのタグ付け、プッシュのステップを含めます。

## フェーズ4: テスト、ドキュメント、その他の改善

10. **テストの実装**:
    *   単体テスト（個々の関数、Pydanticモデル用）および統合テスト（APIエンドポイント用）を開発します。
    *   `pytest` や `httpx` などのツールを利用します。
11. **ロギングとエラーハンドリングの強化**:
    *   詳細なロギングを実装します（`docs/requirements_and_specs.md` に従い標準出力/エラー出力へ）。
    *   OpenAI互換のエラーレスポンスを含む堅牢なエラーハンドリングを保証します。
12. **ドキュメントの強化**:
    *   APIの使用方法、セットアップ、設定、デプロイ手順などを記載した `README.md` を常に最新の状態に保ちます。
    *   FastAPIが自動生成するAPIドキュメント（`/docs`、`/redoc`）を活用します。

この計画はプロジェクトの進行に合わせて更新されます。
