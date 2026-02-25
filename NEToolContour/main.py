import io
import json
import psycopg2
import pandas as pd
import numpy as np
import warnings
import matplotlib
from flask import Flask, send_file, jsonify, Response
from flask_cors import CORS
from scipy.interpolate import griddata  # 确保这行存在
from scipy.interpolate import CloughTocher2DInterpolator

# 1. 基础配置
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning, module='pandas')

app = Flask(__name__)
CORS(app)

DB_CONFIG = {
    "host": "10.222.2.194",
    "port": "5432",
    "database": "dfs_netool",
    "user": "postgres",
    "password": "Landmark1"
}


def get_well_data():
    conn = psycopg2.connect(**DB_CONFIG)
    query = """
    select ws.well_name, ws.latitude, ws.longitude, tubing_head_pressure 
    from public.well_prod_daily wpd
    inner join well_source ws on wpd.well_id = ws.well_id
    where well_name like 'CA%'
    and prod_date = '2025-11-01'
    and latitude is not null
    """
    df = pd.read_sql(query, conn)
    conn.close()
    # 【新增】处理地理位置重复的井，取平均值，防止插值算法崩溃
    df = df.groupby(['longitude', 'latitude', 'well_name']).mean().reset_index()
    return df


@app.route('/api/netool-contour-map')
def generate_2d():
    try:
        df = get_well_data()
        if df.empty:
            return "No data found for the given criteria", 404

        x = df['longitude'].values
        y = df['latitude'].values
        z = df['tubing_head_pressure'].values
        names = df['well_name'].values

        # 网格化
        xi = np.linspace(x.min(), x.max(), 200)
        yi = np.linspace(y.min(), y.max(), 200)
        xi, yi = np.meshgrid(xi, yi)

        # 插值（linear / cubic 都可试，视数据分布选择）
        interp = CloughTocher2DInterpolator(
            (x, y),
            z,
            fill_value=np.nan,  # 或用 z.mean() / 0 等
            rescale=True  # 重要：当经纬度尺度差异大时开启
        )

        zi = interp(xi, yi)

        # ── 绘图 ───────────────────────────────────────────────
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(9, 7), facecolor='#1e1e26')
        ax.set_facecolor('#1e1e26')

        # 等高面填充
        levels = 50
        cntr = ax.contourf(
            xi, yi, zi,
            levels=50,
            cmap="Spectral_r",
            antialiased=True,  # 抗锯齿
            extend='both',
            # 下面两行让颜色过渡更柔和
            alpha=0.95,
            zorder=1
        )

        # 井位置散点（黄色）
        # 可选：再叠加一层平滑的等高线（contour），让轮廓更清晰自然
        ax.contour(
            xi, yi, zi,
            levels=15,  # 少一点，避免太乱
            colors='white',
            linewidths=0.6,
            alpha=0.5,
            linestyles='solid',
            zorder=5
        )

        # 在每个井上标注井名
        for i, name in enumerate(names):
            ax.text(
                x[i], y[i],
                name,
                fontsize=8.5,
                color='white',
                ha='left',
                va='bottom',
                zorder=15,
                bbox=dict(
                    facecolor='black',
                    alpha=0.5,
                    edgecolor='none',
                    pad=1.8,
                    boxstyle='round,pad=0.15'
                ),
                transform=ax.transData
            )

        # 添加颜色条（colorbar）
        cbar = fig.colorbar(cntr, ax=ax, pad=0.03, fraction=0.046, shrink=0.85)
        cbar.set_label('Tubing Head Pressure (unit)', fontsize=10, color='white')
        cbar.ax.tick_params(labelsize=9, colors='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        # 让colorbar的刻度标签和spine颜色为白色（dark模式友好）
        for spine in cbar.ax.spines.values():
            spine.set_color('white')

        # 坐标轴标签与标题
        ax.set_xlabel('Longitude', fontsize=11, color='white')
        ax.set_ylabel('Latitude', fontsize=11, color='white')
        ax.set_title('Tubing Head Pressure Contour - CA* Wells (2025-11-01)',
                     fontsize=13, color='white', pad=12)

        ax.tick_params(axis='both', colors='white', labelsize=9)

        # 紧凑布局
        plt.tight_layout()

        # 保存到内存
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
        img_buf.seek(0)
        plt.close(fig)

        return send_file(img_buf, mimetype='image/png', as_attachment=False)

    except Exception as e:
        import traceback
        return f"Server error: {str(e)}\n{traceback.format_exc()}", 500


@app.route('/api/well-3d-interactive')
def generate_3d_interactive():
    try:
        df = get_well_data()
        df = df.dropna(subset=['longitude', 'latitude', 'tubing_head_pressure'])

        x = df['longitude'].astype(float).values
        y = df['latitude'].astype(float).values
        z = df['tubing_head_pressure'].astype(float).values
        names = df['well_name'].tolist()

        # --- 改进1：增加网格密度 (从50提高到100或150) ---
        grid_size = 100
        xi_list = np.linspace(x.min(), x.max(), grid_size)
        yi_list = np.linspace(y.min(), y.max(), grid_size)
        xm, ym = np.meshgrid(xi_list, yi_list)

        # --- 改进2：使用 cubic (三次) 插值，这是平滑的关键 ---
        # cubic 插值在边缘会产生 NaN，我们用 linear 的结果来填充这些边缘，保证不报错
        zi_cubic = griddata((x, y), z, (xm, ym), method='cubic')
        zi_linear = griddata((x, y), z, (xm, ym), method='linear')

        # 如果 cubic 算不出来的地方（通常是边缘），用 linear 补充
        zi = np.where(np.isnan(zi_cubic), zi_linear, zi_cubic)

        # 最后依然要把残余的 NaN 填充为最小值
        zi_final = np.nan_to_num(zi, nan=float(z.min()))

        result = {
            "data": [
                {
                    "type": "surface",
                    "z": zi_final.tolist(),
                    "x": xi_list.tolist(),
                    "y": yi_list.tolist(),
                    "colorscale": "Spectral",
                    "reversescale": True,
                    # --- 改进3：调整光效，让表面看起来更有质感 ---
                    "lighting": {
                        "ambient": 0.6,
                        "diffuse": 0.5,
                        "fresnel": 0.2,
                        "specular": 0.1,
                        "roughness": 0.5
                    },
                    "contours": {
                        "z": {"show": True, "usecolormap": True, "highlightcolor": "white", "project": {"z": True}}
                    }
                },
                {
                    "type": "scatter3d",
                    "x": x.tolist(), "y": y.tolist(), "z": z.tolist(),
                    "mode": "markers+text",
                    "text": names,
                    "textfont": {"color": "white", "size": 10},
                    "marker": {"size": 5, "color": "white", "opacity": 0.8}
                }
            ],
            "layout": {
            # 1. 使用内置的深色模板
            "template": "plotly_dark",

            # 2. 精确设置背景颜色（#1e1e26 是你 2D 图使用的深蓝色调）
            "paper_bgcolor": "#1e1e26",
            "plot_bgcolor": "#1e1e26",

            "scene": {
                # 3. 设置 3D 空间的背景色
                "bgcolor": "#1e1e26",

                "xaxis": {
                    "title": "Lon",
                    "gridcolor": "#444",      # 网格线颜色
                    "showbackground": True,
                    "backgroundcolor": "#1e1e26" # 坐标轴平面的背景
                },
                "yaxis": {
                    "title": "Lat",
                    "gridcolor": "#444",
                    "showbackground": True,
                    "backgroundcolor": "#1e1e26"
                },
                "zaxis": {
                    "title": "Pressure",
                    "gridcolor": "#444",
                    "showbackground": True,
                    "backgroundcolor": "#1e1e26"
                },
                "aspectmode": "manual",
                "aspectratio": {"x": 1, "y": 1, "z": 0.5}
            },
            "margin": {"l": 0, "r": 0, "b": 0, "t": 30},
            "font": {"color": "white"} # 全局文字改为白色
        }
        }
        return Response(json.dumps(result), mimetype='application/json')

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)