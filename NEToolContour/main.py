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

import plotly.graph_objects as go
@app.route('/api/well-2d-interactive')
def generate_2d_interactive():
    try:
        df = get_well_data()
        if df.empty:
            return jsonify({"error": "No data found"}), 404

        df = df.dropna(subset=['longitude', 'latitude', 'tubing_head_pressure'])

        lon = df['longitude'].astype(float).values
        lat = df['latitude'].astype(float).values
        pressure = df['tubing_head_pressure'].astype(float).values
        names  = df['well_name'].tolist()

        # ── 网格化 ────────────────────────────────────────
        grid_size = 120   # 可调，越大越细腻但计算慢
        xi = np.linspace(lon.min(), lon.max(), grid_size)
        yi = np.linspace(lat.min(), lat.max(), grid_size)
        xm, ym = np.meshgrid(xi, yi)

        # 插值（cubic 更平滑，但边缘易 NaN → 用 linear 补）
        zi_cubic  = griddata((lon, lat), pressure, (xm, ym), method='cubic')
        zi_linear = griddata((lon, lat), pressure, (xm, ym), method='linear')
        zi = np.where(np.isnan(zi_cubic), zi_linear, zi_cubic)
        zi = np.nan_to_num(zi, nan=float(pressure.min()))   # 兜底

        # ── 构建 Plotly data ───────────────────────────────
        contour_data = {
            "type": "contour",
            "x": xi.tolist(),
            "y": yi.tolist(),
            "z": zi.tolist(),
            "colorscale": "Spectral",          # 与 3D 保持一致
            "reversescale": True,
            "contours": {
                "coloring": "fill",             # 填充模式
                "showlabels": True,
                "labelfont": {"color": "white", "size": 10},
                "operation": "=",
                "value": "every 5",             # 每 5 个单位标注一次（可调）
                "showlines": True,
                "line": {"color": "white", "width": 1}
            },
            "line": {"color": "white", "width": 0.8},
            "autocontour": False,               # 关闭自动等高，配合下面手动 levels
            "contours": {                       # 更精细控制
                "start": round(pressure.min() / 5) * 5,
                "end":   round(pressure.max() / 5 + 1) * 5,
                "size":  5
            },
            "showscale": True,
            "colorbar": {
                "title": {"text": "Tubing Head Pressure", "font": {"color": "white"}},
                "tickfont": {"color": "white"}
            }
        }

        # 井点 + 文字标注
        scatter_data = {
            "type": "scatter",
            "x": lon.tolist(),
            "y": lat.tolist(),
            "mode": "markers+text",
            "text": names,
            "textposition": "top center",
            "marker": {
                "size": 8,
                "color": "yellow",
                "line": {"color": "black", "width": 1}
            },
            "textfont": {
                "color": "white",
                "size": 10
            }
        }

        result = {
            "data": [contour_data, scatter_data],
            "layout": {
                "template": "plotly_dark",
                "paper_bgcolor": "#1e1e26",
                "plot_bgcolor":  "#1e1e26",
                "font": {"color": "white"},
                "title": {
                    "text": "Tubing Head Pressure Contour - CA* Wells (2025-11-01)",
                    "font": {"size": 16}
                },
                "xaxis": {
                    "title": "Longitude",
                    "gridcolor": "#444",
                    "zeroline": False
                },
                "yaxis": {
                    "title": "Latitude",
                    "gridcolor": "#444",
                    "zeroline": False,
                    "scaleanchor": "x",    # 保持经纬度比例（重要！）
                    "scaleratio": 1
                },
                "margin": {"l": 50, "r": 50, "t": 60, "b": 50},
                "hovermode": "closest",
                "showlegend": False
            }
        }

        return Response(json.dumps(result), mimetype='application/json')

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

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
        # 强制清理 NaN（防止 contourf 崩溃）
        if np.any(np.isnan(zi)):
            fill_val = np.nanmean(zi) if not np.all(np.isnan(zi)) else 0
            zi = np.nan_to_num(zi, nan=fill_val)

        # ── 绘图部分 ───────────────────────────────────────────────
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 8), facecolor='#1e1e26')  # 稍大一点方便看标签
        ax.set_facecolor('#1e1e26')

        # 1. 填充颜色（底图）
        # 安全获取范围
        vmin, vmax = np.nanmin(zi), np.nanmax(zi)
        if np.isnan(vmin) or np.isnan(vmax):
            return "Interpolation resulted in no valid data", 400
        levels_fill = np.linspace(zi.min(), zi.max(), 60)  # 更多level让颜色过渡更细腻
        cntr_fill = ax.contourf(
            xi, yi, zi,
            levels=levels_fill,
            cmap="Spectral_r",
            antialiased=True,
            extend='both',
            alpha=0.92
        )

        # 2. 绘制等高线（用于标注数字）
        #   levels 可以比填充少一些，避免太密；也可以手动指定想标注的关键值
        manual_levels = np.arange(
            np.round(zi.min() / 5) * 5,
            np.round(zi.max() / 5 + 1) * 5,
            5
        )  # 每5个单位一个主要等高线，例如 0,5,10,...,45

        cntr_lines = ax.contour(
            xi, yi, zi,
            levels=manual_levels,  # 只在这些值上画线并标注
            colors='white',
            linewidths=1.0,
            linestyles='solid',
            alpha=0.9,
            zorder=5
        )

        # 3. 在等高线上添加数字标签（最关键一步）
        labels = ax.clabel(
            cntr_lines,
            levels=manual_levels,  # 只标注这些级别
            inline=True,  # 标签嵌入线内
            inline_spacing=5,  # 标签与线的间距
            fontsize=9,
            colors='white',
            fmt='%.0f',  # 整数显示（可改成 %.1f 显示一位小数）
            rightside_up=True,
            use_clabeltext=True  # 更好的旋转与对齐
        )

        # 让标签有轻微背景（可选，防重叠或底色干扰）
        for label in labels:
            label.set_bbox(dict(
                facecolor='black',
                alpha=0.4,
                edgecolor='none',
                pad=1.2
            ))

        # 4. 井点 + 井名标注（保持不变或微调）
        ax.scatter(x, y, color='yellow', s=50, edgecolor='black', linewidth=1.0, zorder=10)

        for i, name in enumerate(names):
            ax.text(
                x[i], y[i] + 0.00005,  # 轻微上移避免盖住点
                name,
                fontsize=8.5,
                color='white',
                ha='center',  # 改成居中更美观
                va='bottom',
                zorder=15,
                bbox=dict(facecolor='black', alpha=0.55, edgecolor='none', pad=1.6, boxstyle='round,pad=0.2')
            )

        # 5. 颜色条
        cbar = fig.colorbar(cntr_fill, ax=ax, pad=0.04, fraction=0.046, shrink=0.82)
        cbar.set_label('Tubing Head Pressure', fontsize=11, color='white')
        cbar.ax.tick_params(labelsize=9.5, colors='white')
        for spine in cbar.ax.spines.values():
            spine.set_color('white')

        # 轴与标题
        ax.set_xlabel('Longitude', fontsize=11, color='white')
        ax.set_ylabel('Latitude', fontsize=11, color='white')
        ax.set_title('Tubing Head Pressure Contour - CA* Wells (2025-11-01)', fontsize=14, color='white', pad=15)

        ax.tick_params(axis='both', colors='white', labelsize=9.5)

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


@app.route('/api/netool-contour-map-data')
def generate_2d_data():
    try:
        df = get_well_data()
        if df.empty:
            return jsonify({"error": "No data found"}), 404

        # 返回 JSON: 列表形式，便于前端解析
        data = df.to_dict(
            orient='records')  # [{'well_name': 'CA1', 'latitude': 123.45, 'longitude': 67.89, 'tubing_head_pressure': 100}, ...]
        return jsonify({"data": data})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)