### 使うかどうか ―― 結論から

> **結論：最初から main（基礎ブランチ）に薄く入れておくのが吉**
> ──「あとで統合」より **“最初から全実験が自動で残る”** 方がラクです。
> 30 分で済むレベルなら、*exp/ ブランチを切る前* に仕込んでおきましょう。

---

## ✅ MLflow を早めに仕込むメリット

| 何が嬉しい？              | 実際どう活きる？                                     |
| ------------------- | -------------------------------------------- |
| **実験履歴が自動で一元化**     | 日付付きブランチ名＋ハイパラ＋LBスコアをあとで検索しやすい               |
| **モデル/特徴量の比較が即できる** | `mlflow ui` か MLflow Tracking Server でグラフ比較  |
| **再現性**             | 「そのスコアを出すパラメータ・コード・データバージョン」がワンクリック          |
| **チーム or 複数端末で共有**  | S3/GCS などリモートストアに変えてもコードは同じ                  |
| **CI 連携がシンプル**      | Unit test → MLflow → 成功時 tag & push まで自動化できる |

---

## ⚙️ ミニマム実装レシピ（所要 20 分想定）

> **前提**: ローカル or Colab + GitHub Actions + Kaggle API が回る構成

```bash
pip install mlflow==2.12.2  # 例
```

1. **共通 util に 5 行追加**

```python
# utils/mlflow_setup.py
import os, mlflow

def init_mlflow(run_name: str):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment("kaggle_comp")
    return mlflow.start_run(run_name=run_name)
```

2. **train スクリプトのエントリで autolog**

```python
from utils.mlflow_setup import init_mlflow
import mlflow.lightgbm

with init_mlflow(run_name="lgb_depth7_lr03"):
    mlflow.lightgbm.autolog()   # 主要ライブラリは全部 autolog 対応
    model = train_lgb(...)
    mlflow.log_metric("lb_score", 0.803)  # サブミット後に追記でも OK
```

3. **.gitignore に追記**

```
# MLflow
mlruns/
```

4. **GitHub Actions (optional)**

   * pytest → `mlflow artifacts upload` → `kaggle competitions submit -f submission.csv ...`
   * 成功時 `git tag -a sub_${{ steps.date.outputs.today }} -m "LB $SCORE"` など

---

### ブランチ命名との合わせ技

* **main** に MLflow のベース (`mlruns/` は無視)
* 各実験ブランチで `run_name` に *ブランチ名* を入れる → UI で検索一発

  * 例: `exp/20250720_add_age_feat` → run\_name を自動取得 `$(git rev-parse --abbrev-ref HEAD)`

---

## ☝️ ちょっとした注意

| 項目          | Tips                                                     |
| ----------- | -------------------------------------------------------- |
| **ストレージ肥大** | サイズが気になり出したら `MLFLOW_ARTIFACTS_DESTINATION=s3://…` で外出し  |
| **メトリック粒度** | CV スコアと LB スコアを別名でログしておくと後で回帰検証しやすい                      |
| **マルチラン比較** | UI 上で “Parallel Coordinates Plot” が便利。Baseline との差分が一目瞭然 |
| **他ツール併用**  | DVC/Weights & Biases と競合しない。必要なら MLflow → W\&B のブリッジも可   |

---

## まとめ

* **30 分で薄く導入 → `main` 固定がベストプラクティス**
* これで「どのブランチで何点出た？」が迷子にならない
* 将来スケールしても URI を変えるだけでチーム共有に移行できる

> **Go for it!** 明日の実験から自動で履歴が残るので、試行錯誤がぐっと楽になります 🚀
