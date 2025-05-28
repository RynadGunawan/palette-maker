"""
Nama        : Rynad Arkansyah Gunawan
NPM         : 140810230079
Tanggal buat: 25 Mei 2025
Deskripsi   : Website yang dapat menghasilkan color picker berdasarkan warna dominan dari sebuah gambar.
"""

import streamlit as st
import numpy as np
from PIL import Image

# Menambahkan gaya kustom untuk tampilan aplikasi
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    
    /* Menetapkan background dengan efek gradien dan blur */
    .stApp {
        background: linear-gradient(135deg, rgba(245, 247, 250, 0.8) 0%, rgba(195, 207, 226, 0.8) 100%), 
                    url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='100' height='100' viewBox='0 0 100 100'%3E%3Crect width='100' height='100' fill='%23f5f7fa'/%3E%3Cpath d='M0 50 L100 50 M50 0 L50 100' stroke='%23c3cfe2' stroke-width='1' opacity='0.3'/%3E%3C/svg%3E");
        background-size: cover;
        background-attachment: fixed;
        backdrop-filter: blur(15px); 
        max-width: 1200px;
        margin: 0 auto;
        font-family: 'Inter', sans-serif;
    }
    
    /* Gaya untuk header utama */
    .header {
        text-align: center;
        padding: 2rem 0;
        background: rgba(255, 255, 255, 0.7); 
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        margin-bottom: 2rem;
        backdrop-filter: blur(5px);
    }
    
    /* Gaya untuk kontainer umum */
    .container-style {
        background: rgba(255, 255, 255, 0.7); 
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
        backdrop-filter: blur(5px); 
    }
    
    /* Gaya untuk setiap warna dalam palet */
    .color-swatch {
        transition: all 0.3s ease;
        cursor: pointer;
        border-radius: 10px;
        overflow: hidden;
        position: relative;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Efek hover pada color-swatch */
    .color-swatch:hover {
        transform: translateY(-7px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
    }
    
    /* Gaya untuk kontainer preview gambar */
    .image-preview-container {
        background: rgba(255, 255, 255, 0.7);
        border-radius: 15px;
        padding: 1rem;
        margin: 2rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(5px);
    }
</style>
""", unsafe_allow_html=True)

class KMeans:
    def __init__(self, n_clusters=5, max_iter=50, tol=1e-4):
        # Inisialisasi parameter untuk algoritma K-Means
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.counts = None

    def _initialize_centroids(self, X):
        # Menginisialisasi centroid dengan memastikan keunikan warna
        unique_colors = np.unique(X, axis=0)
        if len(unique_colors) >= self.n_clusters:
            return unique_colors[np.random.choice(len(unique_colors), self.n_clusters, replace=False)]
        
        # Menambahkan warna acak jika warna unik kurang dari jumlah cluster
        supplement = X[np.random.choice(len(X), self.n_clusters-len(unique_colors), replace=False)]
        return np.vstack([unique_colors, supplement])

    def fit(self, X):
        # Menginisialisasi centroid awal
        self.centroids = self._initialize_centroids(X)
        
        # Proses iterasi untuk menemukan centroid optimal
        for _ in range(self.max_iter):
            # Menghitung jarak setiap titik data ke centroid
            distances = np.linalg.norm(X[:, None] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            
            # Menghitung centroid baru berdasarkan titik dalam cluster
            new_centroids = []
            for i in range(self.n_clusters):
                cluster_points = X[labels == i]
                if len(cluster_points) == 0:
                    # Menggunakan titik acak jika cluster kosong
                    new_centroids.append(X[np.random.randint(len(X))])
                else:
                    new_centroids.append(cluster_points.mean(axis=0))
            
            new_centroids = np.array(new_centroids)
            
            # Memeriksa konvergensi (perubahan kecil pada centroid)
            if np.max(np.abs(new_centroids - self.centroids)) < self.tol:
                break
                
            self.centroids = new_centroids
        
        # Menghitung jumlah titik dalam setiap cluster
        _, self.counts = np.unique(labels, return_counts=True)
        return self

def process_image(image, resize_dim=None):
    # Mengonversi gambar ke RGB dan mengubah ukuran jika diperlukan
    img = image.convert('RGB')
    if resize_dim:
        # Menghitung ukuran baru dengan mempertahankan aspect ratio
        width, height = img.size
        new_height = int(resize_dim * height/width) if width > height else resize_dim
        new_width = resize_dim if width > height else int(resize_dim * width/height)
        img = img.resize((new_width, new_height))
    return np.array(img)

def main():
    # Menampilkan header utama aplikasi
    st.markdown('<div class="header"><h1>PaletteMaker: Smart Color Extraction with K-Means ✨</h1></div>', 
                unsafe_allow_html=True)
    
    # Menampilkan penjelasan aplikasi dalam layout kolom
    with st.container():
        col1, line_col, col2 = st.columns([2, 0.05, 1])
        with col1:
            st.markdown("""
            ### Discover Your Image's Color Palette
            Upload an image and let our AI-powered color extractor reveal its dominant colors.
            Perfect for designers, artists, or creators!
            """)
            
        with line_col:
            st.markdown(
                """<div style='border-left: 2px solid #e0e0e0; height: 100%; margin: 0 1rem;'></div>""",
                unsafe_allow_html=True
            )
            
        with col2:
            st.markdown("""
            **How it works:**
            1. Upload any JPG/PNG image
            2. Adjust settings
            3. Get your color palette!
            """)

    # Widget untuk mengunggah gambar
    uploaded_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'], 
                                   label_visibility='collapsed')
    
    # Memproses gambar yang diunggah
    if uploaded_file:
        with st.spinner('🔍 Analyzing your image...'):
            try:
                img = Image.open(uploaded_file)
                original_width = img.size[0]
                
                # Pengaturan lanjutan untuk ekstraksi warna
                with st.expander("⚙️ Advanced Settings"):
                    n_colors = st.slider("Number of colors to extract", 3, 10, 5)
                    max_resolution = original_width
                    sample_size = st.slider("Processing resolution", 100, max_resolution, 
                                          min(1000, max_resolution), 
                                          help="Higher resolution = more accurate but slower processing")
                
                # Mengubah ukuran dan memproses gambar
                img_array = process_image(img, sample_size if sample_size < max_resolution else None)
                pixels = img_array.reshape(-1, 3)
                
                # Menampilkan preview gambar
                st.markdown("---")
                st.subheader("Image Preview 🖼")
                preview_img = Image.fromarray(img_array) if sample_size < max_resolution else img
                st.image(preview_img, use_container_width=True, caption=f"Processed at {sample_size}px resolution")
                
                # Ekstraksi warna dominan menggunakan K-Means
                kmeans = KMeans(n_clusters=n_colors).fit(pixels)
                sorted_indices = np.argsort(-kmeans.counts)
                colors = kmeans.centroids[sorted_indices].astype('uint8')
                percentages = (kmeans.counts[sorted_indices] / len(pixels)) * 100

                # Menampilkan palet warna yang diekstraksi
                st.markdown("---")
                st.subheader("Extracted Color Palette 🎨")
                
                cols = st.columns(n_colors)
                color_codes = []
                
                # Menampilkan setiap warna dalam palet
                for i, (col, color, perc) in enumerate(zip(cols, colors, percentages)):
                    with col:
                        hex_code = '#{:02x}{:02x}{:02x}'.format(*color)
                        rgb_code = f'RGB({color[0]}, {color[1]}, {color[2]})'
                        
                        st.markdown(f"""
                        <div class="color-swatch">
                            <div style='
                                background: {hex_code};
                                padding-bottom: 100%;
                                border-radius: 10px;
                            '></div>
                        </div>
                        <div style='text-align: center; margin-top: 0.5rem;'>
                            <div style='
                                font-weight: 600;
                                font-size: 14px;
                                margin-bottom: 4px;
                            '>{hex_code}</div>
                            <div style='
                                font-size: 12px;
                                color: #666;
                            '>{rgb_code}</div>
                            <div style='
                                font-size: 12px;
                                color: #4a90e2;
                                font-weight: 500;
                                margin-top: 4px;
                            '>{perc:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                        color_codes.append(hex_code)

            except Exception as e:
                # Menampilkan pesan error jika terjadi masalah
                st.error(f"⚠️ Error processing image: {str(e)}")

# Menjalankan aplikasi utama
if __name__ == '__main__':
    main()