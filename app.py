import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image
import io

st.set_page_config(
    page_title="Face Classification – JST",
    page_icon="🧠",
    layout="centered",
)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class MLP:
    def __init__(self, w_ih, b_h, w_ho, b_o):
        self.w_ih = w_ih
        self.b_h  = b_h
        self.w_ho = w_ho
        self.b_o  = b_o

    def forward(self, x):
        self.z_h = np.dot(x, self.w_ih) + self.b_h
        self.y_h = sigmoid(self.z_h)
        self.z_o = np.dot(self.y_h, self.w_ho) + self.b_o
        self.y_o = sigmoid(self.z_o)
        return self.y_o

    def predict(self, x):
        out = self.forward(x)
        return int(np.argmax(out)), out.tolist()

def normalize_hu(hu):
    return [-np.sign(v) * np.log10(abs(v) + 1e-10) for v in hu]

def extract_hu_moment(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    moments = cv2.moments(binary)
    hu = cv2.HuMoments(moments).flatten()
    return hu

def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

@st.cache_resource
def load_model():
    try:
        data = joblib.load("mlp_model.joblib")
        w = data["weights"]
        model = MLP(w["w_ih"], w["b_h"], w["w_ho"], w["b_o"])
        scaler = data["scaler"]
        label_map = data.get("label_map", {0: "Human", 1: "Non-Human", 2: "Abu-Abu"})
        return model, scaler, label_map, None
    except FileNotFoundError:
        return None, None, None, "❌ File **mlp_model.joblib** tidak ditemukan. Pastikan file ada di direktori yang sama dengan app.py."

model, scaler, label_map, load_error = load_model()

st.title("🧠 Face Classification – JST MLP")
st.caption("Human · Non-Human · Abu-Abu  |  Berbasis Hu Moments + Sensor Gerak")

if load_error:
    st.error(load_error)
    st.stop()

tab_predict, tab_eval = st.tabs(["📷 Prediksi", "📊 Evaluasi"])

with tab_predict:

    st.subheader("1. Ambil Gambar dari Kamera")
    camera_image = st.camera_input("Arahkan kamera ke wajah, lalu klik ambil gambar")

    st.divider()

    st.subheader("2. Input Sensor Gerak")
    st.caption("Masukkan nilai dari sensor gerak secara manual (0 = tidak ada gerakan, 1 = ada gerakan).")
    movement = st.radio(
        "Status sensor gerak:",
        options=[0, 1],
        format_func=lambda x: "0 – Tidak ada gerakan" if x == 0 else "1 – Ada gerakan",
        horizontal=True,
        key="movement_radio",
    )

    st.divider()

    predict_disabled = camera_image is None
    if predict_disabled:
        st.info("📌 Ambil gambar terlebih dahulu untuk mengaktifkan tombol Predict.")

    predict_btn = st.button(
        "🔍 Predict",
        disabled=predict_disabled,
        use_container_width=True,
        type="primary",
    )

    # ── PROSES PREDIKSI ──
    if predict_btn and camera_image is not None:
        with st.spinner("Mengekstrak fitur & melakukan prediksi…"):
            pil_img = Image.open(io.BytesIO(camera_image.getvalue()))
            bgr_img = pil_to_bgr(pil_img)

            hu = extract_hu_moment(bgr_img)
            hu_norm = normalize_hu(hu)

            feature = np.array(hu_norm + [float(movement)])
            feature_scaled = (feature - scaler["mean"]) / scaler["std"]

            pred_idx, probs = model.predict(feature_scaled)
            pred_label = label_map[pred_idx]

        st.success("✅ Prediksi selesai!")

        col_img, col_result = st.columns([1, 1])

        with col_img:
            st.image(pil_img, caption="Gambar yang diambil", use_container_width=True)

        with col_result:
            st.markdown("### Hasil Prediksi")
            color_map = {"Human": "🟢", "Non-Human": "🔴", "Abu-Abu": "🟡"}
            icon = color_map.get(pred_label, "⚪")
            st.markdown(f"## {icon} **{pred_label}**")

            st.markdown("---")
            st.markdown("**Probabilitas tiap kelas:**")
            for idx, prob in enumerate(probs):
                label = label_map[idx]
                st.progress(float(prob), text=f"{label}: {prob:.4f}")

            st.markdown("---")
            st.caption(f"Sensor gerak: {'Ada gerakan (1)' if movement == 1 else 'Tidak ada gerakan (0)'}")

with tab_eval:
    st.subheader("📊 Evaluasi Model pada Data Test")

    st.markdown(
        """
        Upload file **numpy (.npz)** hasil ekstraksi fitur test yang sudah di-scale.

        File `.npz` harus berisi dua array:
        - `X_test` – fitur test (n_samples × 8), sudah di-normalize
        - `y_test` – label integer (n_samples,), nilai 0/1/2
        """
    )

    uploaded_npz = st.file_uploader("Upload file X_test & y_test (.npz)", type=["npz"])

    if uploaded_npz is not None:
        with st.spinner("Mengevaluasi…"):
            data = np.load(uploaded_npz)
            X_test = data["X_test"]
            y_test = data["y_test"].astype(int)

            y_pred = np.array([model.predict(x)[0] for x in X_test])

            accuracy = np.mean(y_pred == y_test)
            classes = list(label_map.values())
            n = len(classes)

            cm = np.zeros((n, n), dtype=int)
            for t, p in zip(y_test, y_pred):
                cm[t][p] += 1

            precision, recall, f1 = [], [], []
            for i in range(n):
                tp = cm[i][i]
                fp = cm[:, i].sum() - tp
                fn = cm[i, :].sum() - tp
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                precision.append(prec)
                recall.append(rec)
                f1.append(f)

        st.markdown(f"### Akurasi Keseluruhan: **{accuracy * 100:.2f}%**")

        st.markdown("#### Confusion Matrix")
        import pandas as pd
        cm_df = pd.DataFrame(
            cm,
            index=[f"True: {c}" for c in classes],
            columns=[f"Pred: {c}" for c in classes],
        )
        st.dataframe(cm_df, use_container_width=True)

        st.markdown("#### Per-Class Metrics")
        metrics_df = pd.DataFrame({
            "Kelas":     classes,
            "Precision": [f"{p:.4f}" for p in precision],
            "Recall":    [f"{r:.4f}" for r in recall],
            "F1-Score":  [f"{f:.4f}" for f in f1],
        })
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        st.markdown("#### Distribusi Prediksi vs Label Asli")
        dist_df = pd.DataFrame(
            {
                "Label Asli": [int(np.sum(y_test == i)) for i in range(n)],
                "Prediksi":   [int(np.sum(y_pred == i)) for i in range(n)],
            },
            index=classes,
        )
        st.bar_chart(dist_df)

    else:
        st.info("Upload file .npz untuk melihat hasil evaluasi.")

        with st.expander("💡 Cara membuat file .npz dari notebook"):
            st.code(
                """\
# Jalankan di notebook setelah training:
import numpy as np

np.savez(
    "test_data.npz",
    X_test=image_test,                    # sudah di-scale
    y_test=np.argmax(label_test, axis=1)  # konversi one-hot ke integer
)
print("✅ test_data.npz tersimpan")
""",
                language="python",
            )
