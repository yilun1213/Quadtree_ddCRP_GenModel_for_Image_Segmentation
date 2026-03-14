# 実験設定と結果まとめ（`generated_data1`）

## 1. 実験フロー

本実験は、以下の3段階で実施。

1. `config_gen.py` の設定に基づき `generate.py` で合成画像を生成
2. 生成データで `train.py` を実行し、パラメータを推定
3. 推定パラメータを用いて `estimate_label.py` でテスト画像のラベル推定

## 2. 実験設定

## 2.1 データ生成（`config_gen.py` + `generated_data1/true_param`）

- 画像サイズ:
  - $D=2^{d_{\mathrm{max}}}$
  - 本実験では $d_{\mathrm{max}}=8$，したがって $D=256$
- データ数:
  - 学習データ数 $N=100$
  - 評価データ数 $2$（` 100.png`, ` 101.png`）
- ラベル集合:
  - $\mathcal{X}=\{0,1\}$（2クラス）
- 四分木の事前分布 $p(T;\bm{g})$ のパラメータ（真値）:
  - $\bm{g}=(g_0,\dots,g_8)=(0.99,0.9,0.8,0.7,0.6,0.7,0.8,0.9,0)$
- 結合変数の生成分布 $p(\bm{c}\mid T;\alpha,\beta,\eta)$:
  - 親和度関数 $f(s,s')=\exp(\beta B(s,s')+\eta(\mathrm{depth}(s)-\mathrm{depth}(s')))$
  - 設定値 $\alpha=0.001,\beta=0.5,\eta=1.5$
- ラベルの事前分布 $p(x_r;\bm{\omega})$（真値）:
  - 特徴量 $\bm{\phi}(r)=\left(1,\log\mathrm{Area}(r),\log\mathrm{Perimeter}(r),\mathrm{Circularity}(r)\right)$
  - 実装上の feature 名: `log_area`, `log_perimeter`, `circularity`
  - $\bm{\omega}_0=(-0.2,\,0.3,-0.4,0.9)$
  - $\bm{\omega}_1=(0.2,\,-0.3,0.4,-0.9)$
- ピクセル値分布 $p(Y_r\mid x_r;\bm{\theta})$（真値）:
  - 正規分布モデル（RGB）
  - $\bm{\mu}_0=(60,90,160),\ \bm{\mu}_1=(200,170,80)$
  - $\mathrm{std}_0=(20,20,20),\ \mathrm{std}_1=(18,18,18)$
- 実ファイル上の真値パラメータ:
  - `generated_data1/true_param/branch_probs.json`
  - `generated_data1/true_param/label_param.json`
  - `generated_data1/true_param/norm_param.json`

## 2.2 学習（`train.py`）

- 学習データ: `generated_data1/train_data/images`, `generated_data1/train_data/labels`
- データ枚数確認:
  - train images: `100`
  - train labels: `100`
  - test images: `2`
  - test labels: `2`
- 推定出力:
  - 四分木事前分布パラメータの推定値: $\hat{\bm{g}}$
  - ラベル事前分布パラメータの推定値: $\hat{\bm{\omega}}$
  - ピクセル値分布パラメータの推定値: $\hat{\bm{\theta}}$
- 推定ファイル:
  - `generated_data1/estimated_param/branch_probs.json`
  - `generated_data1/estimated_param/label_param.json`
  - `generated_data1/estimated_param/pixel_param.json`

## 2.3 推定（`estimate_label.py`）

- テスト対象: `generated_data1/test_data/images` の2枚（` 100.png`, ` 101.png`）
- 推定手法:
  - 事後分布 $p(T\mid Y)$ に対して $\hat{T}=\arg\max_T p(T\mid Y)$ を計算
  - 固定した $\hat{T}$ の下で，$p(c_s\mid \bm{c}_{-s},\hat{T},Y)$ に基づくギブス更新
  - 最終領域 $R(\hat{\bm{c}})$ 上で
    - $\hat{x}_r=\arg\max_{x\in\mathcal{X}}\left(\log p(x_r;\hat{\bm{\omega}})+\sum_{(i,j)\in r}\log p(y_{(i,j)}\mid x;\hat{\theta}_x)\right)$
  - を用いて $\hat{X}$ を推定
- 反復回数: $M=20$
- バーンイン設定: $B=50$（実装設定）
- OA記録: 各イテレーションで `*_oa_log.txt` に保存

## 3. 実験結果

## 3.1 主要定量結果（OA）

`generated_data1/estimation_results/label/ 100_oa_log.txt` と `generated_data1/estimation_results/label/ 101_oa_log.txt` の両方で、1〜20反復すべて `OA=1.000000`。

| Test image | Iteration 1 OA | Iteration 20 OA | 備考 |
|---|---:|---:|---|
| ` 100.png` | 1.000000 | 1.000000 | 全反復で1.0 |
| ` 101.png` | 1.000000 | 1.000000 | 全反復で1.0 |

## 3.2 真値と推定パラメータの比較（抜粋）

### 四分木事前分布パラメータ $g_d$

| Depth | True | Estimated |
|---:|---:|---:|
| 0 | 0.99 | 1.000000 |
| 1 | 0.90 | 0.890000 |
| 2 | 0.80 | 0.796328 |
| 3 | 0.70 | 0.684548 |
| 4 | 0.60 | 0.594225 |
| 5 | 0.70 | 0.645234 |
| 6 | 0.80 | 0.629762 |
| 7 | 0.90 | 0.474485 |
| 8 | 0.00 | 0.000000 |

### ラベル事前分布パラメータ $\bm{\omega}$

- True: `[[0.3, -0.4, 0.9], [-0.3, 0.4, -0.9]]`
- Estimated: `[[0.2669, -0.3942, 0.5311], [-0.2669, 0.3942, -0.5311]]`

## 4. 可視化結果

ファイル名に先頭空白が含まれるため、Markdownリンクでは `%20` を使用。

### 4.1 Test image ` 100.png`

入力画像:

![input-100](generated_data1/test_data/images/%20100.png)

正解ラベル可視化:

![gt-label-100](generated_data1/test_data/labels/visualize/%20100.png)

推定ラベル可視化（最終）:

![est-label-100](generated_data1/estimation_results/label/visualize/%20100.png)

差分画像（iteration 20）:

![diff-100](generated_data1/estimation_results/label/diff/%20100_0020.png)

真値四分木と推定MAP四分木:

![true-quadtree-100](generated_data1/test_data/quadtrees/%20100.png)
![est-quadtree-100](generated_data1/estimation_results/quadtree/%20100.png)

領域推移（初期とiteration 20）:

![region-100-init](generated_data1/estimation_results/region/%20100_0000.png)
![region-100-final](generated_data1/estimation_results/region/%20100_0020.png)

### 4.2 Test image ` 101.png`

入力画像:

![input-101](generated_data1/test_data/images/%20101.png)

正解ラベル可視化:

![gt-label-101](generated_data1/test_data/labels/visualize/%20101.png)

推定ラベル可視化（最終）:

![est-label-101](generated_data1/estimation_results/label/visualize/%20101.png)

差分画像（iteration 20）:

![diff-101](generated_data1/estimation_results/label/diff/%20101_0020.png)

真値四分木と推定MAP四分木:

![true-quadtree-101](generated_data1/test_data/quadtrees/%20101.png)
![est-quadtree-101](generated_data1/estimation_results/quadtree/%20101.png)

領域推移（初期とiteration 20）:

![region-101-init](generated_data1/estimation_results/region/%20101_0000.png)
![region-101-final](generated_data1/estimation_results/region/%20101_0020.png)
