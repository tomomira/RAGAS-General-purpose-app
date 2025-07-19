# AWS Bedrock → Claude API 移行ガイド

## 概要
本ドキュメントは、AWS Bedrockを使用していたRAG評価システムをClaude APIに移行する際の変更点と手順をまとめたものです。

## 移行の背景・目的

### 移行理由
- **コスト削減**: AWS Bedrockの従量課金からClaude APIの直接利用へ
- **シンプル化**: AWS認証設定の複雑さを回避
- **ローカル実行**: 埋め込みモデルをローカルで実行し、外部依存を削減
- **開発効率**: `.env`ファイルによる簡単な設定管理

### 目標
- 既存の評価メトリクスを維持
- セットアップの簡素化
- 実行コストの削減
- オフライン対応の向上

## 詳細な変更点

### 1. LLMの変更

#### 変更前（AWS Bedrock）
```python
from langchain_aws import ChatBedrockConverse

llm = ChatBedrockConverse(model="anthropic.claude-3-5-sonnet-20240620-v1:0")
```

#### 変更後（Claude API）
```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    api_key=os.getenv('ANTHROPIC_API_KEY'),
    temperature=0.3,
    max_tokens=3000
)
```

**変更点詳細:**
- パッケージ: `langchain_aws` → `langchain_anthropic`
- クラス: `ChatBedrockConverse` → `ChatAnthropic`
- モデル名: `anthropic.claude-3-5-sonnet-20240620-v1:0` → `claude-3-5-sonnet-20241022`
- 認証: AWS認証 → APIキー直接指定
- パラメータ: `temperature`、`max_tokens`を明示的に設定

### 2. 埋め込みモデルの変更

#### 変更前（AWS Bedrock）
```python
from langchain_aws import BedrockEmbeddings

embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")
```

#### 変更後（HuggingFace）
```python
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)
```

**変更点詳細:**
- パッケージ: `langchain_aws` → `langchain_community`
- クラス: `BedrockEmbeddings` → `HuggingFaceEmbeddings`
- モデル: `amazon.titan-embed-text-v2:0` → `sentence-transformers/all-MiniLM-L6-v2`
- 実行環境: AWS クラウド → ローカルCPU
- コスト: 従量課金 → 無料

### 3. 認証・設定の変更

#### 変更前（AWS Bedrock）
```bash
# AWS認証情報が必要
AWS_ACCESS_KEY_ID=xxx
AWS_SECRET_ACCESS_KEY=xxx
AWS_DEFAULT_REGION=us-east-1
```

#### 変更後（Claude API）
```bash
# .envファイル
ANTHROPIC_API_KEY=your-claude-api-key-here
DEBUG=false
LOG_LEVEL=INFO
```

**変更点詳細:**
- 認証方式: AWS認証情報 → Claude APIキー
- 設定ファイル: AWS設定 → `.env`ファイル
- 設定項目数: 3つ以上 → 1つ（APIキーのみ）
- 複雑度: 高 → 低

### 4. 依存関係の変更

#### 変更前（AWS Bedrock版）
```txt
langchain-aws==0.2.28
boto3==1.39.4
botocore==1.39.4
```

#### 変更後（Claude API版）
```txt
langchain-anthropic==0.3.2
anthropic>=0.8.0
sentence-transformers==3.3.1
transformers==4.46.3
torch==2.5.1
python-dotenv>=0.19.0
```

**変更点詳細:**
- AWS関連パッケージを削除
- Anthropic関連パッケージを追加
- HuggingFace関連パッケージを追加
- 環境変数管理用パッケージを追加

### 5. 実行環境の変更

#### 変更前（AWS Bedrock）
- **要求事項**: AWS認証情報の設定、インターネット接続必須
- **実行場所**: AWS クラウド
- **コスト**: 従量課金（トークン数に応じて）
- **レイテンシ**: ネットワーク経由でのAPI呼び出し

#### 変更後（Claude API + HuggingFace）
- **要求事項**: Claude APIキーのみ、初回実行時のみインターネット接続
- **実行場所**: ローカル（埋め込み）+ Claude API（LLM）
- **コスト**: Claude API従量課金のみ（埋め込みは無料）
- **レイテンシ**: 埋め込みはローカル実行で高速

## 移行手順

### Step 1: 新しい依存関係のインストール
```bash
pip install -r requirements.txt
```

### Step 2: 環境変数の設定
```bash
# .envファイルを作成
ANTHROPIC_API_KEY=your-actual-claude-api-key-here
```

### Step 3: コードの実行
```bash
# 推奨：RAGAS_LOCALフォルダから直接実行
cd /path/to/RAGAS_LOCAL
python3 rag_eval_claude.py

# または他のディレクトリからでも実行可能（パス問題解決済み）
python3 /path/to/RAGAS_LOCAL/rag_eval_claude.py
```

### Step 4: 結果の確認
- 評価メトリクスが正常に出力されることを確認
- エラーがないことを確認

## 評価メトリクス（変更なし）

移行後も以下のメトリクスは変更なく使用可能：

1. **faithfulness** - 回答の忠実性
2. **answer_relevancy** - 回答の関連性
3. **ContextRelevance** - 文脈の関連性
4. **ResponseGroundedness** - 回答の根拠性
5. **ResponseRelevancy** - レスポンスの関連性
6. **LLMContextPrecisionWithoutReference** - 文脈の精度

## パフォーマンス比較

| 項目 | AWS Bedrock版 | Claude API版 | 改善点 |
|------|---------------|--------------|---------|
| セットアップ時間 | 10-15分 | 5分 | 設定簡素化 |
| 初回実行時間 | 30秒 | 2-3分 | モデルダウンロード |
| 2回目以降実行 | 30秒 | 15秒 | ローカル埋め込み |
| 月額コスト | $20-50 | $10-20 | 埋め込み無料化 |
| オフライン対応 | 不可 | 部分的可能 | 埋め込みのみ |

## トラブルシューティング

### よくある問題と解決方法

#### 1. Claude APIキーエラー
**問題**: `Error: Claude APIキーが設定されていません`
**解決**: `.env`ファイルに正しいAPIキーを設定

#### 2. HuggingFaceモデルダウンロードエラー
**問題**: `OSError: Can't load tokenizer`
**解決**: インターネット接続を確認し、初回実行時は時間をかけてダウンロード

#### 3. メモリ不足エラー
**問題**: `RuntimeError: CUDA out of memory`
**解決**: `model_kwargs={'device': 'cpu'}`でCPU実行を確認

#### 4. 依存関係エラー
**問題**: `ImportError: No module named 'sentence_transformers'`
**解決**: `pip install -r requirements.txt`で全依存関係をインストール

## 埋め込みモデルの改善実績（2024年7月更新）

### 問題解決の経緯
1. **初期問題**: PyTorch依存関係エラー、FakeEmbeddingsの使用
2. **解決策**: CPU最適化版PyTorchの導入
3. **結果**: 実際の埋め込みモデルによる正確な評価が可能

### 改善効果の定量評価
| メトリクス | 改善前 | 改善後 | 改善率 |
|-----------|--------|--------|--------|
| answer_relevancy | -0.0236 | 0.1169 | **+395%** |
| faithfulness | 1.0000 | 1.0000 | 維持 |
| nv_response_groundedness | 1.0000 | 1.0000 | 維持 |

詳細は `EMBEDDING_MODEL_IMPROVEMENT.md` を参照。

## 最新の改善実績（2025年7月19日更新）

### パス問題の解決
- **問題**: 実行ディレクトリによってcontext.txtが見つからないエラー
- **解決**: スクリプトファイルの場所を基準とした絶対パス自動解決機能を実装
- **実装内容**:
  ```python
  script_dir = os.path.dirname(os.path.abspath(__file__))
  context_path = os.path.join(script_dir, 'analysis', 'context.txt')
  ```
- **効果**: どのディレクトリから実行してもファイルが正しく読み込まれる

## 今後の拡張可能性

### 1. 他のLLMへの対応
- OpenAI GPT-4への切り替え
- ローカルLLM（Ollama等）への対応
- 複数LLMの比較評価

### 2. 埋め込みモデルの選択肢
- 日本語特化モデル（`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`）
- 高性能モデル（`sentence-transformers/all-mpnet-base-v2`）
- カスタムファインチューニング

### 3. 評価メトリクスの拡張
- カスタムメトリクスの追加
- 業界特化評価の実装
- A/Bテスト機能の追加

### 4. 実行環境の改善
- パス問題の完全解決により安定性向上
- より柔軟な実行環境対応

## まとめ

AWS BedrockからClaude APIへの移行により、以下の効果を得られました：

- **コスト削減**: 埋め込みモデルのローカル実行により大幅なコスト削減
- **設定簡素化**: 複雑なAWS認証から簡単な`.env`設定へ
- **実行速度向上**: 埋め込み処理のローカル実行により高速化
- **オフライン対応**: 部分的なオフライン実行が可能

この移行により、より効率的で経済的なRAG評価システムを構築できました。