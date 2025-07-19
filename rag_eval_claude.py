# 外部ライブラリをインポート
import os

# スクリプトファイルの場所を基準にした絶対パスを使用する
# どのディレクトリから実行してもcontext.txtが見つかるようにするため
script_dir = os.path.dirname(os.path.abspath(__file__))
context_path = os.path.join(script_dir, 'analysis', 'context.txt')

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic

from datasets import Dataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    # RAG 基本メトリクス
    answer_relevancy,           # 質問に対する回答の関連度（LLM）
    faithfulness,               # 回答が文脈に忠実か（LLM）
    context_precision,          # 取得文脈の純度（LLM/非LLM）
    context_recall,             # 必要文脈の取りこぼし率（LLM/非LLM ground_truth 必要）
    context_entity_recall,      # 正解エンティティが文脈が編集している割合（ground_truth 必要）
    # 回答 vs 参照
    answer_correctness,         # 回答の正確性（LLM）
    answer_similarity,          # 意味的類似度（Embeddings）
    RougeScore,                 # ROUGE-L F1
    BleuScore,                  # BLEU-4 ※短文や日本語は0になりやすい
    SemanticSimilarity,         # Cosine 類似度（Embeddings）
    ExactMatch,                 # 文字列完全一致
    FactualCorrectness,         # 事実正確性（LLM）
    # NVIDIA-Judge
    AnswerAccuracy,             # LLM 2段階評価の回答精度
    ContextRelevance,           # 文脈が質問に関連しているか
    ResponseGroundedness,       # 回答が文脈に根ざす度合い
    ResponseRelevancy,          # 回答系統の関連度評価（LLM）
    # 派生
    LLMContextPrecisionWithReference,       # LLM判定版 Precision（参照必須）
    LLMContextPrecisionWithoutReference,    # LLM判定版 Precision（参照なし）
    LLMContextRecall,                       # LLM判定版 Recall（参照必須）
    NonLLMContextPrecisionWithReference,    # 埋め込み距離版 Precision（参照必須）
    NonLLMContextRecall,                    # 埋め込み距離版 Recall（参照必須）
    NonLLMStringSimilarity,                 # 非LLM文字列類似度（Levenshtein など）
    NoiseSensitivity,                       # ノイズ混入耐性（LLM）
)

# 環境変数の読み込み
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("警告: python-dotenvライブラリがインストールされていません。")

# LLMと埋め込みモデルを設定
llm = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    api_key=os.getenv('ANTHROPIC_API_KEY'),
    temperature=0.3,
    max_tokens=3000,
    max_retries=5,  # リトライ回数を設定
)

# HuggingFaceの埋め込みモデルを使用（CPU最適化版）
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

# 検索対象の「社内文書」を作成
"""
texts = [
    "情報1: KDDIアジャイル開発センター株式会社は、KAGという略称で親しまれています。",
    "情報2: KAG社は、かぐたんというSlackアプリを開発しました。",
]

texts = [
    "情報1: ソニーは家庭用ゲーム機「PlayStation」シリーズを開発・販売している。",
    "情報2: PlayStation 5は、2020年11月に発売された第5世代のモデルである。",
    "情報3: PlayStation 5の主要な特徴として、超高速SSDによるロード時間の短縮や、DualSenseワイヤレスコントローラーによるハプティックフィードバックが挙げられる。",
]

texts = [
    "情報1: 株式会社エナリスは、法人向けと新電力事業者向けにエネルギーソリューションを提供する企業です。",
    "情報2: 同社は、企業の脱炭素化を支援する「eneGX」ブランドや、小規模設備を統合制御するVPPプラットフォームサービスを提供しています。",
    "情報3: 株式会社エナリスは、電力の需給管理に関する長年のノウハウを持ち、脱炭素社会の実現に貢献しています。"
]
"""

# 検索対象の「社内文書」をファイルから読み込む
"""
try:
    with open('00_Inbox/一時保存/Ragas/RAGAS_LOCAL/analysis/context.txt', 'r', encoding='utf-8') as f:
        # レポートを1行ずつ分割してリストに格納
        texts = f.read().splitlines()
except FileNotFoundError:
    print("エラー: 'context.txt' が見つかりません。")
    # サンプルデータで続行
    #texts = [
    #    "異常検知レポート 日付: 2024-07-16",
    #    "ID: panel_015, 項目: 予測発電量, 異常値: 150.5 kWh, 詳細: 予測値が閾値(120.0 kWh)を大幅に超過。",
    #    "ID: panel_032, 項目: 乖離値(予測-実績), 異常値: -35.2 kWh, 詳細: 実績が予測を大幅に上回り、乖離が閾値(±30.0 kWh)を超過。",
    #    "ID: panel_088, 項目: 予測発電量, 異常値: 5.1 kWh, 詳細: 日照条件に対して予測値が異常に低く、機器不具合の可能性あり。",
    #    "ID: panel_091, 項目: 乖離値(予測-実績), 異常値: 40.1 kWh, 詳細: 予測が実績を大幅に上回り、乖離が閾値(±30.0 kWh)を超過。"
    #]
"""
try:
    with open(context_path, 'r', encoding='utf-8') as f:
        # レポートを1行ずつ分割してリストに格納
        texts = f.read().splitlines()
except FileNotFoundError:
    print(f"エラー: '{context_path}' が見つかりません。")
    # サンプルデータで続行
    texts = [
        "情報1: 株式会社エナリスは、法人向けと新電力事業者向けにエネルギーソリューションを提供する企業です。",
        "情報2: 同社は、企業の脱炭素化を支援する「eneGX」ブランドや、小規模設備を統合制御するVPPプラットフォームサービスを提供しています。",
        "情報3: 株式会社エナリスは、電力の需給管理に関する長年のノウハウを持ち、脱炭素社会の実現に貢献しています。"
    ]


# ベクトルDBをローカルPC上に作成
vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings)

# RAGの検索対象としてベクトルDBを指定
retriever = vectorstore.as_retriever()

# プロンプトテンプレートを定義
prompt = ChatPromptTemplate.from_template(
    "背景情報をもとに質問に回答してください。背景情報：{context} 質問：{question}"
)

# RAGを使ったLangChainチェーンを定義
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 質問文を入れてチェーンを実行
#question = "かぐたんって何？"
#question = "PlayStation 5の主な特徴を教えてください。"
question = "株式会社エナリスとその主なサービスや貢献について説明してください。"


answer = chain.invoke(question)

# LLMからの出力を表示
print(answer)

# 自動Ground Truth生成関数
def generate_ground_truth(question, contexts, llm):
    prompt = f"""以下の文脈情報を使用して、質問に対する理想的な回答を1-2文で作成してください。

文脈：{contexts}
質問：{question}

回答は正確で簡潔にしてください。専門用語は適切に使用し、文脈から導き出せる情報のみを含めてください。
"""
    response = llm.invoke(prompt)
    return response.content

# Ground Truth自動生成
auto_ground_truth = generate_ground_truth(question, texts, llm)
print(f"\n自動生成されたGround Truth: {auto_ground_truth}")


# 評価対象のデータセットを定義（自動生成Ground Truth使用）
dataset = Dataset.from_dict({
    "question": [question],
    "answer": [answer],
    "contexts": [texts],
    "ground_truth": [auto_ground_truth],
})

"""
# Ground Truthを手動で設定
manual_ground_truth = "株式会社エナリスは、法人や新電力向けにエネルギー関連サービスを提供する企業で、脱炭素化支援の「eneGX」やVPPプラットフォームが主なサービスです。"
print(f"\n手動設定されたGround Truth: {manual_ground_truth}")

# 評価対象のデータセットを定義（手動設定Ground Truth使用）
dataset = Dataset.from_dict({
    "question": [question],
    "answer": [answer],
    "contexts": [texts],
    "ground_truth": [manual_ground_truth],  # ← ここで手動設定した変数を渡す
})
"""

# 評価を実行（Ground Truth対応メトリクスを含む）
result = evaluate(
    dataset,
    metrics=[
        # === 基本メトリクス ===
        faithfulness,        # 忠実性：背景情報と一致性のある回答ができているか
        answer_relevancy,    # 関連性：質問と関連した回答ができているか
        
        # NVIDIA-Judge系（内部で列名が自動マッピングされる）
        ContextRelevance(),      # 文脈が質問に関連しているか
        ResponseGroundedness(),  # 回答が文脈に根ざす度合い
        ResponseRelevancy(),     # レスポンスの関連性
        
        # Context系
        LLMContextPrecisionWithoutReference(),  # LLM判定版Context Precision（参照不要版）
        
        # === Ground Truth対応メトリクス（自動生成Ground Truth使用）===
        answer_correctness,     # 正確性：Ground Truthとの一致度
        answer_similarity,      # 意味的類似度：Ground Truthとの類似性
        context_precision,      # 文脈精度：Ground Truthに関連した文脈の取得精度
        context_recall,         # 文脈再現：必要な文脈の取りこぼし率
        RougeScore(),          # ROUGE-L F1スコア
        SemanticSimilarity(),   # コサイン類似度
        FactualCorrectness(),   # 事実正確性
    ],
    llm=LangchainLLMWrapper(llm),
    embeddings=LangchainEmbeddingsWrapper(embeddings),
)

# その他のGround Truth対応メトリクス（必要に応じて追加可能）
 #BleuScore(),            # BLEU-4スコア（短文では0になりやすい）
 #ExactMatch(),           # 完全一致評価
 #context_entity_recall,  # エンティティ回収率

# 評価結果の詳細表示
print("\n=== 評価結果詳細 ===")
# EvaluationResultオブジェクトからPandas DataFrameに変換してからメトリクスを取得
df = result.to_pandas()
metrics_summary = df.mean(numeric_only=True)
for metric, score in metrics_summary.items():
    print(f"{metric}: {score:.4f}")