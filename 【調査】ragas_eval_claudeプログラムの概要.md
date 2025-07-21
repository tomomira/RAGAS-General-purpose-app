# 【調査】ragas_eval_claudeプログラムの概要

## プログラム概要

このプログラムは、**RAG（Retrieval-Augmented Generation）システムの評価**を行うための構造になっています。以下のような段階的な構成です：

## 1. 初期設定部分（1-70行）

```python
# スクリプトファイルの場所を基準とした絶対パス設定（2-7行）
script_dir = os.path.dirname(os.path.abspath(__file__))
context_path = os.path.join(script_dir, 'analysis', 'context.txt')

# ライブラリインポート
- LangChain関連（RAGチェーン構築用）
- RAGAS評価メトリクス（複数の評価指標）
- HuggingFace埋め込み、Claude LLM
```

**目的**: 柔軟なパス処理と必要なライブラリと評価メトリクスを読み込み

## 2. LLM・埋め込みモデル設定（57-70行）

```python
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", ...)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", ...)
```

**目的**: 評価に使用するLLMと埋め込みモデルを初期化

## 3. 検索対象データ作成（109-124行）

```python
# context.txtファイルからのデータ読み込み（自動パス解決）
try:
    with open(context_path, 'r', encoding='utf-8') as f:
        texts = f.read().splitlines()
except FileNotFoundError:
    # フォールバック用のサンプルデータ
    texts = ["情報1: 株式会社エナリス...", ...]

vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings)
retriever = vectorstore.as_retriever()
```

**目的**: 外部ファイルまたはサンプルデータでベクトルDBを構築（パス問題解決済み）

## 4. RAGチェーン構築（83-92行）

```python
prompt = ChatPromptTemplate.from_template("背景情報をもとに...")
chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
```

**目的**: 質問→検索→生成の一連のRAGパイプライン作成

## 5. 実際の質問実行（93-98行）

```python
question = "かぐたんって何？"
answer = chain.invoke(question)
```

**目的**: RAGシステムで実際に質問に回答させる

## 6. Ground Truth自動生成（99-107行）

```python
def generate_ground_truth(question, contexts, llm):
    # LLMに理想的な回答を生成させる
```

**目的**: 評価用の正解データをLLMで自動生成

## 7. 評価データセット作成（108-115行）

```python
dataset = Dataset.from_dict({
    "question": [question],
    "answer": [answer],
    "contexts": [texts],
    "ground_truth": [auto_ground_truth],
})
```

**目的**: RAGAS評価用のデータセット形式に整形

## 8. 評価実行（116-148行）

```python
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, ...],
    llm=LangchainLLMWrapper(llm),
    embeddings=LangchainEmbeddingsWrapper(embeddings),
)
```

**目的**: 複数の評価メトリクスでRAGシステムの性能を測定

## 9. 結果表示（149-161行）

```python
df = result.to_pandas()
metrics_summary = df.mean(numeric_only=True)
for metric, score in metrics_summary.items():
    print(f"{metric}: {score:.4f}")
```

**目的**: 評価結果を数値で表示

## プログラムの特徴

- **自動評価**: Ground Truthを手動作成せずLLMで自動生成
- **多角的評価**: 忠実性、関連性、正確性など複数の観点で評価
- **完全ローカル**: FAISSベクトルDBとローカル埋め込みモデル使用
- **柔軟なパス処理**: スクリプトファイルの場所を基準とした自動パス解決
- **実用的**: 実際のRAGシステムの評価パイプラインとして使用可能
- **ポータブル**: どのディレクトリから実行してもファイルアクセスが正常動作

## プログラムの最大のメリット：Ground Truth自動生成

### 1. 従来の課題解決

**従来の問題**
- Ground Truthの手動作成が困難・コストが高い
- Ground Truth必須のメトリクスが使用できない
- 評価可能なメトリクス数が限定的（3-4個程度）

**このプログラムの解決策**
- **LLMによる自動Ground Truth生成**
- **Ground Truth必須メトリクスが使用可能に**
- **評価メトリクス数の大幅拡張**（12個以上）

### 2. 評価メトリクスの分類と拡張

#### Ground Truth不要メトリクス（従来から使用可能）
```python
# 基本メトリクス
faithfulness,                    # 忠実性：背景情報との一致性
answer_relevancy,               # 関連性：質問と回答の関連度

# NVIDIA-Judge系
ContextRelevance(),             # 文脈の質問関連性
ResponseGroundedness(),         # 回答の文脈根拠性
ResponseRelevancy(),            # レスポンス関連性

# Context系
LLMContextPrecisionWithoutReference(),  # 参照不要版Context Precision
```

#### Ground Truth必須メトリクス（自動生成により使用可能）
```python
# 正確性・類似性評価
answer_correctness,             # Ground Truthとの一致度
answer_similarity,              # 意味的類似度
SemanticSimilarity(),          # コサイン類似度
FactualCorrectness(),          # 事実正確性

# 文脈評価
context_precision,              # 文脈取得精度
context_recall,                # 文脈取りこぼし率

# 言語学的評価
RougeScore(),                  # ROUGE-L F1スコア
# BleuScore(),                 # BLEU-4スコア（参考）
# ExactMatch(),                # 完全一致評価（参考）
```

### 3. 実用例：異常値検出システムでの活用

**質問例**: 「XX月●●日に異常値はありますか？」

**従来の課題**
- 期間データから正確なGround Truthを人手で作成する必要
- 異常値の判定基準を事前に定義する必要

**このプログラムでの解決**
1. **Context**: 期間内の正常・異常データを投入
2. **自動分析**: Claudeがcontextデータを分析
3. **Ground Truth生成**: 「XX月●●日には異常値が3件検出されています」等
4. **包括的評価**: 12個以上のメトリクスで多角的評価

### 4. 革命的なメリット

| 項目 | 従来手法 | このプログラム |
|------|----------|----------------|
| **Ground Truth作成** | 人手で時間をかけて作成 | LLMが自動生成 |
| **使用可能メトリクス** | 3-4個（Ground Truth不要のみ） | 12個以上（包括的） |
| **評価の多角性** | 限定的 | 忠実性・正確性・類似性・文脈精度など |
| **運用コスト** | 高い（人手作業） | 低い（自動化） |
| **評価速度** | 遅い（事前準備必要） | 高速（リアルタイム評価） |

**最大のメリット**: **Ground Truth自動生成**により、本来Ground Truthが必要で使用できなかった評価メトリクスが利用可能となり、**より多角的で包括的なRAG評価**が実現できること

## Ground Truth関連の処理

### 1. 自動Ground Truth生成箇所

```python
def generate_ground_truth(question, contexts, llm):
    prompt = f"""以下の文脈情報を使用して、質問に対する理想的な回答を1-2文で作成してください。

文脈：{contexts}
質問：{question}

回答は正確で簡潔にしてください。専門用語は適切に使用し、文脈から導き出せる情報のみを含めてください。
"""
    response = llm.invoke(prompt)
    return response.content
```

この関数では、LLMに対して質問と文脈を渡し、理想的な回答（ground truth）を生成するよう指示しています。

### 2. Ground Truth対応メトリクスでの確認箇所

以下のメトリクスは内部でLLMを使用してground truthとの比較評価を行います：

- **`answer_correctness`**: LLMが生成された回答とground truthの正確性を比較
- **`context_precision`**: LLMがground truthに関連した文脈の取得精度を評価
- **`context_recall`**: LLMが必要な文脈の取りこぼし率をground truthと比較して評価
- **`FactualCorrectness()`**: LLMが回答の事実正確性をground truthと照合

### 3. 評価実行時のLLM設定

評価時に指定されたLLM（Claude 3.5 Sonnet）が、各メトリクスの評価時にground truthとの比較を行います。

つまり、LLMは**生成時**（ground truth作成）と**評価時**（メトリクス計算）の両方でground truthに関与しています。

## まとめ

このようにRAGシステムの**構築→実行→評価**の全工程を一つのプログラムで実現する構造になっています。特に、Ground Truthの自動生成により、手動での正解データ作成作業を省力化している点が特徴的です。 