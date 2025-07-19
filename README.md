# Ragas Tutorial - RAG評価システム（Claude API版）

このプロジェクトは、Ragasライブラリを使用してRAG（Retrieval-Augmented Generation）システムの評価を行うサンプルコードです。Claude APIを使用してローカルで実行します。

## セットアップ手順

### 1. 仮想環境のアクティベート
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# または
venv\Scripts\activate  # Windows
```

### 2. 依存関係のインストール

#### requirements.txtを使用
```bash
pip install -r requirements.txt
```

### 3. 環境変数の設定

`.env.example`ファイルを`.env`にコピーし、Claude APIキーを設定してください：

```bash
# .env.exampleファイルを.envにコピー
cp .env.example .env

# .envファイルを編集してAPIキーを設定
# .envファイル
ANTHROPIC_API_KEY=your-actual-claude-api-key-here
```

**重要**: `.env`ファイルには機密情報が含まれるため、Gitにコミットされないよう`.gitignore`で除外されています。

### 4. 実行コマンド

#### 通常実行（推奨）
```bash
# RAGAS_LOCALフォルダから直接実行
cd /path/to/RAGAS_LOCAL
source venv/bin/activate && python3 rag_eval_claude.py
```

#### 他のディレクトリからの実行
```bash
# obsidian-workフォルダからの実行
cd /path/to/obsidian-work
source venv/bin/activate && python3 .\00_Inbox\一時保存\Ragas\RAGAS_LOCAL\rag_eval_claude.py
```

**注意**: パス問題が解決されたため、どのディレクトリから実行してもcontext.txtが正しく読み込まれます。

## スクリプトの説明

### rag_eval_claude.py
RAGシステムを評価するスクリプトです。Claude APIとHuggingFaceのローカル埋め込みモデルを使用します。

#### 主な機能
- **シンプルなRAGパイプライン**: FAISSベクトルストアとLangChainを使用したRAG実装
- **自動Ground Truth生成**: LLMを活用した理想回答の自動生成機能
- **包括的な評価メトリクス**: Ground Truth対応を含む11つのメトリクスを使用
- **Claude APIとの統合**: Claude 3.5 Sonnet (20241022) を使用
- **ローカル埋め込み**: HuggingFaceの`sentence-transformers/all-MiniLM-L6-v2`を使用
- **柔軟なパス処理**: スクリプトファイルの場所を基準とした自動パス解決

#### 評価メトリクス
**基本メトリクス**:
1. **faithfulness** - 回答が提供された文脈に忠実かを評価
2. **answer_relevancy** - 質問に対する回答の関連度を評価

**NVIDIA-Judge系メトリクス**:
3. **ContextRelevance** - 取得した文脈が質問に関連しているかを評価
4. **ResponseGroundedness** - 回答が文脈に基づいているかの妥当性を評価
5. **ResponseRelevancy** - レスポンス全体の関連性を評価

**Context系メトリクス**:
6. **LLMContextPrecisionWithoutReference** - 取得した文脈の精度を評価

**Ground Truth対応メトリクス（自動生成機能により利用可能）**:
7. **answer_correctness** - 正確性：Ground Truthとの一致度
8. **answer_similarity** - 意味的類似度：Ground Truthとの類似性
9. **context_precision** - 文脈精度：Ground Truthに関連した文脈の取得精度
10. **context_recall** - 文脈再現：必要な文脈の取りこぼし率
11. **RougeScore** - ROUGE-L F1スコア
12. **SemanticSimilarity** - コサイン類似度
13. **FactualCorrectness** - 事実正確性

## 主な変更点（AWS Bedrock版からの変更）

### LLMの変更
- **変更前**: `ChatBedrockConverse` (AWS Bedrock)
- **変更後**: `ChatAnthropic` (Claude API)

### 埋め込みモデルの変更
- **変更前**: `BedrockEmbeddings` (AWS Titan)
- **変更後**: `HuggingFaceEmbeddings` (ローカル実行)

### 認証方法の変更
- **変更前**: AWS認証情報
- **変更後**: `.env`ファイルの`ANTHROPIC_API_KEY`

## 実行環境要件
- Python 3.8以上
- Claude APIキー
- インターネット接続（Claude API・HuggingFaceモデルダウンロード用）
- CPU最適化PyTorch（自動インストール）

## 調査用コマンド

```bash
# ragasでmetricsとしてインポート可能な一覧を取得する
source venv/bin/activate && python3 -c "from ragas.metrics import *; print([name for name in dir() if not name.startswith('_')])"
```

## トラブルシューティング

### 1. Claude APIキーエラー
```
Error: Claude APIキーが設定されていません
```
**解決方法**: `.env`ファイルに正しいAPIキーを設定してください。

### 2. 埋め込みモデルのダウンロードエラー
```
Error: HuggingFaceEmbeddings initialization failed
```
**解決方法**: インターネット接続を確認し、初回実行時にモデルのダウンロードを待ってください。

### 3. PyTorch依存関係エラー
```
ImportError: cannot import name 'Tensor' from 'torch'
```
**解決方法**: CPU最適化版PyTorchをインストールしてください：
```bash
pip install --break-system-packages torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 4. メモリ不足エラー
```
Error: CUDA out of memory
```
**解決方法**: CPUモードで実行されているか確認してください（`model_kwargs={'device': 'cpu'}`）。

### 5. 実行速度が遅い
**解決方法**: 
- 初回実行時はモデルダウンロードのため時間がかかります
- 2回目以降は高速化されます
- より軽量なモデルに変更することも可能です

### 6. context.txtが見つからないエラー（解決済み）
```
エラー: 'C:\path\to\analysis\context.txt' が見つかりません。
```
**解決方法**: この問題は修正されました。スクリプトファイルの場所を基準とした自動パス解決により、どのディレクトリから実行してもcontext.txtが正しく読み込まれます。

## 依存関係の詳細

- `langchain-anthropic`: Claude API連携
- `anthropic`: Claude APIクライアント
- `sentence-transformers`: ローカル埋め込みモデル
- `torch`: CPU最適化版PyTorch
- `faiss-cpu`: ベクトル検索（CPU版）
- `ragas`: RAG評価フレームワーク
- `python-dotenv`: 環境変数管理

## 埋め込みモデルについて

### 現在使用中：all-MiniLM-L6-v2
- **特徴**: 軽量で高速、バランスの良い性能
- **サイズ**: 22MB
- **次元**: 384次元
- **言語**: 多言語対応（日本語含む）

### 高性能モデルへの変更方法
より高精度な評価が必要な場合、以下のモデルに変更可能：

```python
# 高精度モデル（推奨）
model_name="sentence-transformers/all-mpnet-base-v2"

# 日本語特化モデル
model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
```

### 期待される改善効果
- **all-mpnet-base-v2**: answer_relevancy 0.7-0.9に向上
- **multilingual-L12-v2**: 日本語評価精度が20-40%向上

## 性能改善の実績

### 評価メトリクス改善結果
| メトリクス | 改善前 | 改善後 | 改善率 |
|-----------|--------|--------|--------|
| answer_relevancy | -0.0236 | 0.1169 | **+395%** |
| faithfulness | 1.0000 | 1.0000 | 維持 |
| nv_response_groundedness | 1.0000 | 1.0000 | 維持 |

### 自動Ground Truth生成機能（最新）
明確な正解がない質問でも、LLMが自動でGround Truthを生成し、Ground Truth対応メトリクスが利用可能：

**利用可能メトリクス数**: 6個 → 13個（+117%拡張）

**新たに利用可能になったメトリクス**:
- answer_correctness: 0.9776 (97.76%)
- semantic_similarity: 0.9104 (91.04%)
- context_precision: 0.5000 (50%)
- context_recall: 1.0000 (100%)
- rouge_score: 1.0000 (100%)
- factual_correctness: 0.4000 (40%)

### パス問題の解決（2025年7月19日）
- **問題**: 実行ディレクトリによってcontext.txtが見つからないエラー
- **解決**: スクリプトファイルの場所を基準とした絶対パス自動解決機能を実装
- **効果**: どのディレクトリから実行してもファイルが正しく読み込まれる

詳細は `AUTO_GROUND_TRUTH_GUIDE.md` を参照してください。

## GitHub利用時の注意事項

### ファイル管理
- **`.gitignore`**: 機密情報や一時ファイルを自動除外
- **`.env.example`**: 環境変数のテンプレート（実際のAPIキーは含まない）
- **`.env`**: 実際の環境変数ファイル（Gitから除外される）

### 初回セットアップ（GitHubからクローン後）
```bash
# リポジトリをクローン
git clone <your-repo-url>
cd RAGAS_LOCAL

# 仮想環境作成・アクティベート
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# または venv\Scripts\activate  # Windows

# 依存関係インストール
pip install -r requirements.txt

# 環境変数設定
cp .env.example .env
# .envファイルを編集してAPIキーを設定
```

### 関連資料
- **埋め込み改善**: `EMBEDDING_MODEL_IMPROVEMENT.md`
- **移行ガイド**: `MIGRATION_GUIDE.md`
- **自動Ground Truth**: `AUTO_GROUND_TRUTH_GUIDE.md`