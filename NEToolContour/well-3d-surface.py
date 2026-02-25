import io
import psycopg2
import pandas as pd
import numpy as np
import warnings
import matplotlib

# 核心：设置非交互后端
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 导入3D绘图工具
from scipy.interpolate import griddata
from flask import Flask, send_file
from flask_cors import CORS

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
    return df


@app.route('/api/well-3d-surface')
def generate_3d_plot():
    try:
        df = get_well_data()
        if df.empty: return "No data", 404

        x = df['longitude'].values
        y = df['latitude'].values
        z = df['tubing_head_pressure'].values

        # 1. 插值生成网格数据 (3D 表面必须基于网格)
        xi = np.linspace(x.min(), x.max(), 100)
        yi = np.linspace(y.min(), y.max(), 100)
        xi, yi = np.meshgrid(xi, yi)
        zi = griddata((x, y), z, (xi, yi), method='cubic')

        # 2. 绘图设置
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(12, 9), facecolor='#1e1e26')
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('#1e1e26')

        # 3. 绘制 3D 曲面
        # cmap 可以换成 'magma', 'viridis', 'Spectral_r'
        surf = ax.plot_surface(xi, yi, zi, cmap='Spectral_r',
                               edgecolor='none', alpha=0.8, antialiased=True)

        # 4. 绘制底部的等值线投影 (让图看起来更专业)
        ax.contourf(xi, yi, zi, zdir='z', offset=np.nanmin(zi) - 0.5, cmap='Spectral_r', alpha=0.3)

        # 5. 绘制原始井点（悬浮在曲面上或对应的位置）
        ax.scatter(x, y, z, color='white', s=40, edgecolors='black', depthshade=True)

        # 6. 标注井号
        for i, txt in enumerate(df['well_name']):
            ax.text(x[i], y[i], z[i] + 0.1, txt, size=8, color='yellow', fontweight='bold')

        # 7. 设置视角和标签
        ax.view_init(elev=30, azim=225)  # 调整观察角度：仰角30度，方位角225度
        ax.set_xlabel('Longitude', color='gray')
        ax.set_ylabel('Latitude', color='gray')
        ax.set_zlabel('Pressure', color='gray')
        ax.set_title('3D Well Pressure Surface', color='white', pad=20, fontsize=15)

        # 添加颜色条
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Pressure (MPa)')

        # 8. 保存导出
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', dpi=130, bbox_inches='tight')
        img_buf.seek(0)
        plt.close(fig)

        return send_file(img_buf, mimetype='image/png')

    except Exception as e:
        print(f"3D Error: {e}")
        return str(e), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)