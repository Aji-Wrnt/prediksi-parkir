import pickle
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

with open("model_parkir.pkl", "rb") as file:
    model = pickle.load(file)


def hitung_hari_paling_kosong():
    nama_hari = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu"]
    total_kosong_per_hari = {}
    for i in range(6):
        X_test = pd.DataFrame({"hari": [i]})
        prediksi = model.predict(X_test)
        total_kosong_per_hari[nama_hari[i]] = int(np.sum(prediksi == 0))
    hari_terbanyak = max(total_kosong_per_hari, key=total_kosong_per_hari.get)
    return {
        "hari_terbanyak": hari_terbanyak,
        "jumlah_slot": total_kosong_per_hari[hari_terbanyak],
    }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    day = int(data.get("day", 0))

    # Bungkus input menjadi DataFrame sebelum dimasukkan ke model.predict()
    X_test = pd.DataFrame({"hari": [day]})
    prediction_flatten = model.predict(X_test)

    # Ubah kembali array 10 elemen menjadi matriks visual 2 baris x 5 kolom untuk Frontend
    prediction_matrix = prediction_flatten.reshape(2, 5).tolist()
    tren_kosong = hitung_hari_paling_kosong()

    return jsonify({"layout": prediction_matrix, "tren": tren_kosong})


if __name__ == "__main__":
    app.run(debug=True)
