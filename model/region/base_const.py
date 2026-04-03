# -*- coding: utf-8 -*-
"""
領域モデルの基本定数・設定

この module は現在使用されていません。
代わりに affinity.py の親和度関数を使用してください。

今後、複数の領域生成戦略（CRPベース、その他のベイズノンパラメトリック法など）
を实装する場合には、このモジュールを拡張する予定です。
"""

# 使用例：異なる親和度関数の選択
# import model.region.affinity as affinity_func
# 
# # 親和度関数の選択：
# # - affinity_boundary_and_depth: 論文 2.4 の例 2（推奨）
# # - affinity_boundary_only: シンプル版
# # - affinity_constant: テスト用
#
# selected_affinity = affinity_func.affinity_boundary_and_depth
# affinity_params = {"beta": 1.0, "eta": 0.5}
