from flask import Flask, render_template, request, send_from_directory, redirect, url_for
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 避免沒有 GUI
import matplotlib.pyplot as plt
import seaborn as sns
import os
import zipfile
import uuid

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ===== 首頁 =====
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part"
        file = request.files["file"]
        if file.filename == "":
            return "No selected file"

        file_id = str(uuid.uuid4())
        filename = f"{file_id}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        output_subfolder = os.path.join(OUTPUT_FOLDER, file_id)
        os.makedirs(output_subfolder, exist_ok=True)

        # 根據勾選決定是否加入分類
        include_l2 = request.form.get("include_l2") == "on"

        generate_boxplots(filepath, output_subfolder, include_l2)

        # 打包 ZIP
        zip_path = os.path.join(OUTPUT_FOLDER, f"{file_id}_boxplots.zip")
        with zipfile.ZipFile(zip_path, 'w') as zf:
            for root, _, files in os.walk(output_subfolder):
                for f in files:
                    zf.write(os.path.join(root, f), arcname=f)

        return redirect(url_for('download_file', filename=f"{file_id}_boxplots.zip"))

    return render_template("index.html")

# ===== 下載 ZIP =====
@app.route("/download/<filename>")
def download_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)

# ===== 產生箱型圖 =====
def generate_boxplots(file_path, output_folder, include_l2=False):
    df = pd.read_excel(file_path)

    if include_l2:
        # 每組別 + L2分類
        for group_name, group_data in df.groupby("表格1.Group"):
            n_categories = group_data["分類L2"].nunique()
            fig_width = max(10, n_categories * 0.5)
            plt.figure(figsize=(fig_width, 6))

            upper_bounds = []
            for cat, subdata in group_data.groupby("分類L2"):
                y = subdata["GPMS重量(g)"].dropna()
                if len(y) > 0:
                    Q1 = y.quantile(0.25)
                    Q3 = y.quantile(0.75)
                    IQR = Q3 - Q1
                    upper_bounds.append(Q3 + 1.5 * IQR)
            if not upper_bounds:
                continue

            y_max = max(upper_bounds)
            y_min = 0
            y_range = y_max - y_min
            y_max += 0.05 * y_range

            sns.boxplot(
                data=group_data,
                x="分類L2",
                y="GPMS重量(g)",
                palette="Set3",
                width=0.6,
                fliersize=2
            )
            plt.ylim(y_min, y_max)
            plt.title(f"Group {group_name}", fontsize=14)
            plt.xlabel("Category L2", fontsize=12)
            plt.ylabel("GPMS Weight (g)", fontsize=12)
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, f"{group_name}_boxplot.png"), dpi=300)
            plt.close()
    else:
        # 每組別單張圖
        for i, (group_name, group_data) in enumerate(df.groupby("表格1.Group"), start=1):
            values = group_data["GPMS重量(g)"].dropna()
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            y_min = max(values.min(), lower)
            y_max = min(values.max(), upper)
            y_range = y_max - y_min
            y_min -= 0.1 * y_range
            y_max += 0.1 * y_range

            plt.figure(figsize=(5, 6))
            sns.boxplot(
                data=group_data,
                y="GPMS重量(g)",
                color="lightblue",
                width=0.3,
                fliersize=2
            )
            plt.ylim(y_min, y_max)
            plt.ylabel("GPMS Weight (g)", fontsize=12)
            plt.title(f"Group {i}", fontsize=14)
            plt.xticks([])
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, f"group_{i}_boxplot.png"), dpi=300)
            plt.close()

# ===== 啟動 Flask =====
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5500, debug=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
