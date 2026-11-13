import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Eigenfaces Reconstruction Slider", layout="wide")

DATA_PATH = Path(__file__).parent / "olivetti_faces.npz"


@st.cache_resource
def load_model():
    data = np.load(DATA_PATH)
    X = data["X"].astype(np.float64)                 # (400, 4096)
    subject_ids = data["subject_ids"]
    n = X.shape[0]

    mu = X.mean(axis=0)                              # mean face
    X_c = X - mu                                     # centered

    C = (X_c @ X_c.T) / n                            # snapshot matrix (400, 400)
    vals, vecs = np.linalg.eigh(C)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]

    keep = n - 1                                     # last eigenvalue is ~0
    U_raw = X_c.T @ vecs[:, :keep]                   # (4096, 399)
    U = U_raw / np.linalg.norm(U_raw, axis=0)        # unit-length eigenfaces
    lam = np.clip(vals[:keep], 1e-12, None)          # eigenvalues (variances)

    Z = X_c @ U                                      # codes for all faces
    return X, np.asarray(subject_ids), mu, X_c, U, lam, Z


X, subject_ids, mu, X_c, U, lam, Z = load_model()


# ---------- 2) Controls ----------
st.title("🧑 Eigenfaces: Reconstruction Slider")
st.write("Drag the slider to choose how many eigenfaces (k) are used to rebuild the face.")

st.sidebar.header("Controls")
k = st.sidebar.slider("Rank k (number of eigenfaces)", min_value=1, max_value=100, value=30)
face_id = int(st.sidebar.number_input("Face number (0-399)", min_value=0, max_value=399,
                                      value=100, step=1))


# ---------- 3) Compute for the chosen face and k ----------
x = X[face_id]
z = Z[face_id, :k]                                   # k numbers describing this face
x_rec = z @ U[:, :k].T + mu                          # rebuild

loss_pct = 100 * np.sum((x - x_rec) ** 2) / np.sum(X_c[face_id] ** 2)
d2 = np.sum(z ** 2 / lam[:k])                        # Mahalanobis d² using k components

top3 = np.argsort(np.abs(z))[::-1][:3]               # strongest eigenfaces for this face


# ---------- 4) KPI cards ----------
c1, c2 = st.columns(2)
c1.metric("Reconstruction Error (% Loss)", f"{loss_pct:.2f}%")
c2.metric("Mahalanobis Distance Score (d²)", f"{d2:.1f}",
          delta=f"{d2 - k:+.1f} vs. typical face (about {k})", delta_color="off")

st.caption(f"Storing {k} numbers instead of 4096 → {4096 / k:.0f}x smaller. "
           f"This is photo #{face_id} (person #{subject_ids[face_id]}).")


# ---------- 5) Pictures: original | reconstructed | top 3 eigenfaces ----------
def to_display(img_vec):
    """Stretch any vector to 0..1 so we can draw it as an image."""
    a = img_vec.reshape(64, 64)
    return (a - a.min()) / (a.max() - a.min() + 1e-12)

fig, axes = plt.subplots(1, 5, figsize=(15, 3.4))

axes[0].imshow(x.reshape(64, 64), cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Original")

axes[1].imshow(x_rec.reshape(64, 64), cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"Reconstructed (k = {k})")

for j in range(3):
    ax = axes[2 + j]
    if j < len(top3):
        i = top3[j]
        ax.imshow(to_display(U[:, i]), cmap="gray")
        ax.set_title(f"Eigenface #{i + 1}\nweight {z[i]:+.2f}")
    # if k < 3 there are fewer than 3 eigenfaces, so the extra slots stay empty

for ax in axes:
    ax.axis("off")

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

with st.expander("What am I looking at?"):
    st.write(
        "- **Original**: the real photo (4096 numbers).\n"
        "- **Reconstructed**: rebuilt from only k numbers: mean face + weighted eigenfaces.\n"
        "- **Top 3 eigenfaces**: the building blocks this face leans on most. "
        "Positive weight adds the pattern, negative weight subtracts it.\n"
        "- **% Loss**: share of this face's detail that is missing after compression.\n"
        "- **Mahalanobis d²**: how unusual this face's k numbers are, measured in units "
        "of normal wiggle. For typical faces it is about k."
    )