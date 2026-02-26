# 修改后的 /api/netool-contour-map 路由
# 注意: 需要先安装 Plotly: pip install plotly
# 导入 Plotly (在文件顶部添加)
import plotly.graph_objects as go

@app.route('/api/well-2d-interactive')
def generate_2d_interactive():
    try:
        df = get_well_data()
        if df.empty:
            return jsonify({"error": "No data found"}), 404

        x = df['longitude'].values
        y = df['latitude'].values
        z = df['tubing_head_pressure'].values
        names = df['well_name'].values

        # 网格化 (类似于3D的准备)
        grid_size = 200  # 提高密度以获得更平滑的等值线
        xi_list = np.linspace(x.min(), x.max(), grid_size)
        yi_list = np.linspace(y.min(), y.max(), grid_size)
        xm, ym = np.meshgrid(xi_list, yi_list)

        # 插值 (使用cubic以获得平滑效果，类似于3D)
        zi_cubic = griddata((x, y), z, (xm, ym), method='cubic')
        zi_linear = griddata((x, y), z, (xm, ym), method='linear')
        zi = np.where(np.isnan(zi_cubic), zi_linear, zi_cubic)
        zi_final = np.nan_to_num(zi, nan=float(z.min()))

        # 构建 Plotly data
        data = [
            {
                "type": "contour",
                "z": zi_final.tolist(),
                "x": xi_list.tolist(),
                "y": yi_list.tolist(),
                "colorscale": "Spectral_r",  # 匹配原Matplotlib的cmap
                "reversescale": False,  # 根据需要调整
                "contours": {
                    "coloring": "heatmap",  # 或 'fill' / 'lines'，'heatmap'类似于contourf
                    "showlines": True,  # 显示等值线
                    "start": zi_final.min(),
                    "end": zi_final.max(),
                    "size": (zi_final.max() - zi_final.min()) / 60  # 类似于levels_fill
                },
                "line": {
                    "smoothing": 1.3,  # 平滑线条
                    "width": 1.0
                },
                "colorbar": {
                    "title": "Tubing Head Pressure",
                    "titleside": "right",
                    "tickfont": {"color": "white"}
                }
            },
            {
                "type": "scatter",
                "x": x.tolist(),
                "y": y.tolist(),
                "mode": "markers+text",
                "text": names,
                "textposition": "top center",  # 类似于原text偏移
                "marker": {
                    "color": "yellow",
                    "size": 10,
                    "line": {"color": "black", "width": 1}
                },
                "textfont": {"color": "white", "size": 10}
            }
        ]

        # 构建 layout (匹配原Matplotlib的暗黑主题)
        layout = {
            "template": "plotly_dark",
            "paper_bgcolor": "#1e1e26",
            "plot_bgcolor": "#1e1e26",
            "title": {
                "text": "Tubing Head Pressure Contour - CA* Wells (2025-11-01)",
                "font": {"color": "white", "size": 14}
            },
            "xaxis": {
                "title": "Longitude",
                "tickfont": {"color": "white"},
                "gridcolor": "#444"
            },
            "yaxis": {
                "title": "Latitude",
                "tickfont": {"color": "white"},
                "gridcolor": "#444"
            },
            "margin": {"l": 50, "r": 50, "b": 50, "t": 50},
            "font": {"color": "white"},
            "hovermode": "closest"  # 启用鼠标交互显示数据
        }

        result = {"data": data, "layout": layout}
        return Response(json.dumps(result), mimetype='application/json')

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)