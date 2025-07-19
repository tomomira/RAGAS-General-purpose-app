---
marp: true
theme: default
class: lead
paginate: true
backgroundColor: #fff
---

# 【調査】ragas_eval_claudeプログラムの概要

**RAGシステム評価プログラムの詳細分析**

---

## プログラム概要

このプログラムは、**RAG（Retrieval-Augmented Generation）システムの評価**を行うための構造になっています。

**特徴**
- 段階的な構成でRAGシステムを構築・評価
- Ground Truthの自動生成
- 複数メトリクスによる多角的評価

---

## 初期構成（1～3ステップ）

### 1. 初期設定部分（1-61行）
- LangChain関連ライブラリ
- RAGAS評価メトリクス
- HuggingFace埋め込み、Claude LLM

### 2. LLM・埋め込みモデル設定（62-72行）
- Claude 3.5 Sonnet設定
- HuggingFace埋め込みモデル初期化

### 3. 検索対象データ作成（73-82行）
- 社内文書模擬データ作成
- FAISSベクトルDB構築

---

## RAG実行（4～6ステップ）

### 4. RAGチェーン構築（83-92行）
```python
chain = ({"context": retriever, "question": RunnablePassthrough()} 
         | prompt | llm | StrOutputParser())
```

### 5. 実際の質問実行（93-98行）
```python
question = "かぐたんって何？"
answer = chain.invoke(question)
```

### 6. Ground Truth自動生成（99-107行）
- LLMによる理想的回答の自動生成
- 評価用正解データ作成

---

## 評価実行（7～9ステップ）

### 7. 評価データセット作成（108-115行）
- RAGAS評価用データセット形式に整形

### 8. 評価実行（116-148行）
- 複数メトリクスでの性能測定
- LLMとEmbeddingsを使用した評価

### 9. 結果表示（149-161行）
- 評価結果の数値表示
- メトリクス別スコア出力

---

## プログラムの特徴

### 🤖 **自動評価**
Ground Truthを手動作成せずLLMで自動生成

### 📊 **多角的評価** 
忠実性、関連性、正確性など複数の観点で評価

### 💻 **完全ローカル**
FAISSベクトルDBとローカル埋め込みモデル使用

### ⚡ **実用的**
実際のRAGシステムの評価パイプラインとして使用可能

---

## Ground Truth処理：自動生成

```python
def generate_ground_truth(question, contexts, llm):
    prompt = f"""以下の文脈情報を使用して、質問に対する
    理想的な回答を1-2文で作成してください。

文脈：{contexts}
質問：{question}

回答は正確で簡潔にしてください。"""
    response = llm.invoke(prompt)
    return response.content
```

**目的**: LLMが質問と文脈から理想的な回答を自動生成

---

## Ground Truth処理：評価メトリクス

### LLMによる比較評価メトリクス

- **`answer_correctness`**: 生成回答とground truthの正確性比較
- **`context_precision`**: ground truthに関連した文脈取得精度
- **`context_recall`**: 必要文脈の取りこぼし率評価
- **`FactualCorrectness()`**: 回答の事実正確性照合

### 評価実行時の仕組み
評価時にClaude 3.5 Sonnetが各メトリクス計算時にground truthとの比較を実行

---

## Ground Truth処理：LLMの二重役割

### 🏗️ **生成時の役割**
- Ground Truth作成
- 理想的回答の自動生成

### ⚖️ **評価時の役割**  
- メトリクス計算
- Ground Truthとの比較評価

**重要**: LLMが**生成**と**評価**両方でground truthに関与

---

## まとめ

### RAGシステムの完全評価パイプライン
**構築 → 実行 → 評価** の全工程を一つのプログラムで実現

### 主な革新点
- **Ground Truth自動生成**: 手動正解データ作成の省力化
- **多角的評価**: 複数メトリクスによる総合的性能評価  
- **実用性**: 実際のRAGシステム開発で即利用可能

### 応用可能性
社内文書検索、Q&Aシステム、チャットボットなどの評価に活用可能

---

## 🚀 最大のメリット：Ground Truth自動生成

### 従来の課題
- ❌ **手動作成コスト**: Ground Truth作成に時間とコストが必要
- ❌ **メトリクス制限**: Ground Truth必須メトリクスが使用不可
- ❌ **評価の限界**: 使用可能メトリクス数が3-4個程度

### このプログラムの革新
- ✅ **LLM自動生成**: Claude が理想的回答を自動作成
- ✅ **メトリクス拡張**: Ground Truth必須メトリクスが使用可能
- ✅ **包括的評価**: 12個以上のメトリクスで多角的評価

---

## 📊 評価メトリクスの分類と拡張

### Ground Truth **不要** メトリクス（従来から使用可能）
```
faithfulness               # 忠実性：背景情報との一致性
answer_relevancy          # 関連性：質問と回答の関連度
ContextRelevance()        # 文脈の質問関連性
ResponseGroundedness()    # 回答の文脈根拠性
ResponseRelevancy()       # レスポンス関連性
LLMContextPrecisionWithoutReference()  # 参照不要版
```

**従来使用可能**: 約6個のメトリクス

---

## 📈 評価メトリクスの分類と拡張

### Ground Truth **必須** メトリクス（自動生成により使用可能）
```
answer_correctness        # Ground Truthとの一致度
answer_similarity         # 意味的類似度
context_precision         # 文脈取得精度
context_recall           # 文脈取りこぼし率
SemanticSimilarity()     # コサイン類似度
FactualCorrectness()     # 事実正確性
RougeScore()            # ROUGE-L F1スコア
```

**新たに使用可能**: 約7個のメトリクス

---

## 🎯 実用例：異常値検出システム

### シナリオ
**質問**: 「XX月●●日に異常値はありますか？」  
**Context**: 期間内の正常・異常データ

### 従来手法の課題
- 📝 人手でGround Truth作成が必要
- ⏱️ 異常値判定基準の事前定義が必要
- 💰 高い作業コスト

### このプログラムの解決
1. **Context投入**: 期間データを自動で処理
2. **Claude分析**: データパターンを自動解析
3. **Ground Truth生成**: 「XX月●●日には異常値が3件検出」
4. **包括評価**: 12個以上のメトリクスで評価

---

## ⚡ 革命的な改善効果

| 項目 | 従来手法 | このプログラム |
|------|----------|----------------|
| **Ground Truth作成** | 👥 人手で時間をかけて作成 | 🤖 LLMが自動生成 |
| **使用可能メトリクス** | 📊 3-4個（限定的） | 📈 12個以上（包括的） |
| **評価の多角性** | ⚠️ 限定的 | ✅ 忠実性・正確性・類似性・文脈精度 |
| **運用コスト** | 💸 高い（人手作業） | 💰 低い（自動化） |
| **評価速度** | 🐌 遅い（事前準備必要） | ⚡ 高速（リアルタイム） |

---

## 🎉 Ground Truth自動生成の価値

### 🏆 **最大のメリット**
**Ground Truth自動生成**により、本来Ground Truthが必要で使用できなかった評価メトリクスが利用可能となり、**より多角的で包括的なRAG評価**が実現

### 🌟 **ゲームチェンジャー**
- RAG評価の**運用コスト大幅削減**
- **リアルタイム包括評価**の実現
- **評価品質の向上**と**開発効率化**の両立

---

## Thank You

**質問・ディスカッションをお待ちしています** 