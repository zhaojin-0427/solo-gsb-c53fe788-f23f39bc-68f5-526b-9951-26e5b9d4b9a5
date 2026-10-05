# 包围曝光合成 HTTP API

用 Python + FastAPI + OpenCV + NumPy 实现的包围曝光（exposure bracketing）合成服务。
接收 3～9 张同尺寸 **16 位线性 RGB PNG**，依据**未剪切的静态背景**做整数平移配准，
容忍高对比度移动物体（不牵引位移、不产生重影），输出 **float32 RGB 辐亮度 TIFF**
及运动/未知掩膜。

## 目录结构

```
app/
  config.py        固定参数（位移上限、配准块尺寸、纹理/一致性阈值等）
  models.py        /merges 请求与响应模型、字段校验
  image_io.py      Base64 / 16 位 PNG / float32 TIFF 编解码
  registration.py  未剪切静态背景上的块级整数平移配准
  fusion.py        运动检测、辐亮度融合、掩膜生成
  main.py          FastAPI 入口（POST /merges、GET /health）
tests/             pytest 合成场景测试（19 个）
scripts/           生成 README 示例的脚本
examples/          真实请求/响应（由脚本生成）
Dockerfile, docker-compose.yml
```

## 一键启动（Docker Compose）

```bash
docker compose up --build
```

服务监听 `http://localhost:8000`。健康检查：

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

也可本地直接运行（Python 3.11）：

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 接口

### `POST /merges`

**请求体（JSON）**

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `frames` | string[3..9] | 每帧为同尺寸 **16 位线性 RGB PNG** 的标准 Base64；任一通道为 `0` 或 `65535` 的像素视为无效（剪切/饱和） |
| `exposures` | number[3..9] | 各帧**有限正**曝光秒数，数量与 `frames` 相同 |
| `reference_index` | integer | 参考帧序号，**0 起始**，必须在帧范围内 |
| `tolerance` | number | **有限非负**辐亮度容差 |

**响应体（JSON，200）**

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `radiance_tiff` | string | **float32 RGB TIFF** 的 Base64；辐亮度 = `有效采样 / 65535 / 曝光秒数`，未知像素为 `NaN` |
| `motion_mask_png` | string | 8 位灰度 PNG 的 Base64，运动像素 `255`，其余 `0` |
| `unknown_mask_png` | string | 8 位灰度 PNG 的 Base64，未知像素 `255`，其余 `0` |
| `shifts` | object[] | 每帧 `{"dy": int, "dx": int}`，顺序与输入一致，参考帧为 `{0,0}`，每轴绝对值 ≤ 8 |

**位移语义**：`(dy, dx)` 是采样偏移——注册后参考坐标 `(y, x)` 的内容取自该帧
`(y+dy, x+dx)`；即把源帧反向平移 `(-dy, -dx)` 即与参考帧重合。

### 处理规则（与实现一致）

1. **输入校验**：帧数 3～9；曝光为有限正数；`reference_index` 合法；`tolerance` 有限非负；
   所有 PNG 必须为 3 通道 16 位且尺寸相同。非法请求返回 `422`。
2. **配准**：只在参考帧与候选帧**都未剪切**的像素上比较灰度辐亮度
   （`mean(raw)/65535/曝光秒数`）。在参考帧上均匀选取有纹理的背景块，每块在
   `dy,dx ∈ [-8,8]` 上做块匹配，再对所有块的最优位移做直方图投票，并用全部背景块的
   25 分位残差复核。移动物体只占据少数块、无法主导票数，故不会牵引位移；
   背景块不足、票数不一致或远处候选残差相当（无法确认静态背景）时返回 `422`。
3. **运动检测**：对齐后，对任两帧都有效的同坐标像素，若**任一通道**辐亮度差
   `> tolerance`，标记为运动区。剪切像素不参与比较。
4. **融合**：
   - 静态像素：对所有**有效观测**的辐亮度做曝光时长加权平均；
   - 运动像素：**只取参考帧**的有效辐亮度，因此移动物体不会重影；
   - 运动区参考观测无效、静态区所有帧观测都无效（含配准后越界）→ **未知**，
     辐亮度写 `NaN` 并置位未知掩膜。

## 实际请求 / 响应示例

`examples/request.json` 与 `examples/response.json` 是用当前代码真实生成的一对
请求/响应（128×128，3 帧，曝光 0.25/1.0/4.0 秒，含高对比度移动圆盘）。
可复现：启动服务后执行 `python scripts/generate_example.py`。

发起请求：

```bash
curl -s -X POST http://localhost:8000/merges \
  -H 'Content-Type: application/json' \
  --data @examples/request.json -o examples/response.json
```

该示例的实际响应（Base64 字段很长，已截断）：

```json
{
  "radiance_tiff": "SUkqAAgAAAASAP4ABAABAAAAAwAAAAMAAGBAAA...（float32 RGB TIFF 的 Base64）",
  "motion_mask_png": "iVBORw0KGgoAAAANSUhEUgAAAIAAAACACAYAA...（运动掩膜 PNG 的 Base64）",
  "unknown_mask_png": "iVBORw0KGgoAAAANSUhEUgAAAIAAAACACAYAA...（未知掩膜 PNG 的 Base64）",
  "shifts": [
    {"dy": -3, "dx": 4},
    {"dy": 0, "dx": 0},
    {"dy": 5, "dx": -2}
  ]
}
```

对应摘要（完整值见 `examples/summary.txt`）：

```text
radiance TIFF: shape=(128, 128, 3), dtype=float32, min=0.0100, max=0.6000
motion pixels:  1058
unknown pixels: 0
shifts: [{'dy': -3, 'dx': 4}, {'dy': 0, 'dx': 0}, {'dy': 5, 'dx': -2}]
```

用 Python 解码并检查结果：

```python
import base64, json, cv2, numpy as np

resp = json.load(open("examples/response.json"))

def decode_png(b64):
    return cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8),
                        cv2.IMREAD_UNCHANGED)

radiance = cv2.imdecode(
    np.frombuffer(base64.b64decode(resp["radiance_tiff"]), np.uint8),
    cv2.IMREAD_UNCHANGED)          # float32 BGR；OpenCV 按 BGR 写 TIFF
radiance = cv2.cvtColor(radiance, cv2.COLOR_BGR2RGB)
motion = decode_png(resp["motion_mask_png"])     # uint8，运动=255
unknown = decode_png(resp["unknown_mask_png"])   # uint8，未知=255
print(radiance.shape, radiance.dtype, resp["shifts"])
# (128, 128, 3) float32 [{'dy': -3, 'dx': 4}, {'dy': 0, 'dx': 0}, {'dy': 5, 'dx': -2}]
```

### 错误示例

输入非法或背景无法配准时返回 HTTP `422`，例如帧数不是 3～9：

```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": ["body", "frames"],
      "msg": "Value error, 帧数必须在 3~9 之间",
      "input": ["..."]
    }
  ]
}
```

背景无法依据未剪切静态背景配准时：

```json
{"detail": "背景无法配准: 第 2 帧背景块无法就位移达成一致（最高票数 2 < 4），无法配准"}
```

## 测试

```bash
pip install pytest httpx
pytest -q
```

测试覆盖：真实位移恢复（移动前景不牵引位移）、静态背景辐亮度精度、运动区只取
参考帧、无重影、运动区参考无效/静态区全无效/越界 → 未知，以及各类非法输入
（帧数、曝光、容差、序号、尺寸、位深、Base64）和背景不可配准的 422 行为。
