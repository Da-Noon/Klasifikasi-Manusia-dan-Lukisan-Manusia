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
    import pandas as pd

    st.subheader("📊 Evaluasi Proses Training")

    # ── Data loss per epoch (dari hasil training) ──
    epoch_losses = [
        0.368467, 0.200634, 0.151817, 0.137913, 0.132482,
        0.129711, 0.128063, 0.126950, 0.126122, 0.125514,
        0.125019, 0.124608, 0.124266, 0.123923, 0.123720,
        0.123548, 0.123350, 0.123174, 0.122974, 0.122872,
        0.122664, 0.122665, 0.122520, 0.122429, 0.122438,
        0.122278, 0.122258, 0.122152, 0.122138, 0.122053,
        0.122026, 0.121952, 0.121929, 0.121887, 0.121846,
        0.121791, 0.121763, 0.121636, 0.121689, 0.121695,
        0.121630, 0.121602, 0.121567, 0.121547, 0.121507,
        0.121507, 0.121452, 0.121424, 0.121352, 0.121381,
    ]
    epochs = list(range(1, len(epoch_losses) + 1))

    # ── Metrik ringkas ──
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Epoch", len(epochs))
    col2.metric("Loss Awal", f"{epoch_losses[0]:.6f}")
    col3.metric("Loss Akhir", f"{epoch_losses[-1]:.6f}",
                delta=f"{epoch_losses[-1] - epoch_losses[0]:.6f}",
                delta_color="inverse")

    st.divider()

    # ── Grafik Loss per Epoch ──
    st.markdown("#### 📉 Kurva Loss Training")
    loss_df = pd.DataFrame({"Loss": epoch_losses}, index=epochs)
    loss_df.index.name = "Epoch"
    st.line_chart(loss_df, use_container_width=True)

    st.divider()

    # ── Tabel lengkap epoch & loss ──
    st.markdown("#### 📋 Tabel Loss per Epoch")
    table_df = pd.DataFrame({
        "Epoch": epochs,
        "Loss":  [f"{v:.6f}" for v in epoch_losses],
        "Δ Loss": ["–"] + [
            f"{epoch_losses[i] - epoch_losses[i-1]:+.6f}"
            for i in range(1, len(epoch_losses))
        ],
    })
    st.dataframe(table_df, use_container_width=True, hide_index=True, height=300)
