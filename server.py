import os

import numpy as np
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
import shutil
import judge_slope_danger_grad_cam
import count_pic
import itertools
print("サーバを立ち上げ中")
# ./flask_api_index.htmlからPOSTされた画像を一時保存するディレクトリ
UPLOAD_FOLDER = "../images/predict_img/original"
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
def index():
    return render_template("./flask_api_index.html")


@app.route("/result", methods=["POST"])
def result():
    if request.method == "POST":
        # "upload_files"はflask_api_index.htmlから選択されたファイル
        if request.files.getlist("upload_files")[0].filename:
            # originalディレクトリを一旦削除
            shutil.rmtree("../images/predict_img/original")
            # originalディレクトリを作成(要はoriginalディレクトリを空にした)
            os.makedirs("../images/predict_img/original")
            # flask_api_index.htmlから選択されたファイル数を計測
            upload_files = request.files.getlist("upload_files")
            len_upload_files = len(upload_files)

            for upload_file in upload_files:
                # 受信したファイルをoriginal/に保存
                upload_file.save(
                    "../images/predict_img/original/"
                    + secure_filename(upload_file.filename)
                )
                # /tmpフォルダのファイル数をカウント
                count_file = count_pic.count_pic()
                # おまじない(ないとupload_file.saveが何故か出来ない)
                upload_file.stream.seek(0)
                # 受信したファイルを連番をつけてtmp/に保存
                upload_file.save(
                    "../judge/tmp/"
                    + count_file
                    + secure_filename(upload_file.filename)
                )
            # judge_slope_danger_grad_dam.allはoriginal/ディレクトリにある画像を読みこみ出力する。出力結果をjudgedへ代入
            judged = judge_slope_danger_grad_cam.all()
            # numpy配列なのでリストに変換
            path_judged_l2d = judged[0].tolist()
            # リストのリストなのでさらに平坦化（1次元のリスト）する
            path_judged = list(itertools.chain.from_iterable(path_judged_l2d))
            # 判明結果の画面でファイル名表示用のリストを作成(filenameをrender_templateにわたす)
            filename = []
            fp_index = 0
            for i in path_judged:
                fn = os.path.basename(path_judged[fp_index])
                filename.append(fn)
                fp_index += 1
            # 判定結果の画面のイルカ(サメ)の根拠は～の部分で使うpredict_resultの値を小数点第2位まで丸める。
            rounded_pr = np.round(judged[1], decimals=2)
        # あらかじめ用意した"./result.html"に変数を渡す
        return render_template(
            "./result.html",
            rounded_pr=rounded_pr,
            judged=judged[2],
            len_upload_files=len_upload_files,
            filelist=path_judged,
            upload_files=upload_files,
            filename=filename,
            heatmap_name=judged[5]
        )


# 画像表示用
@app.route("/images/predict_img/original/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    app.debug = True
    # 開発中はhost="localhost",社内で公開する場合はhost="0.0.0.0"
    app.run(host="localhost", port=5000)
