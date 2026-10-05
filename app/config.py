"""包围曝光合成服务的固定参数。"""

# 每轴允许的最大整数平移（像素）
MAX_SHIFT = 8

# 配准：背景块（贴片）尺寸与搜索边距（边距 = MAX_SHIFT，保证任意位移下贴片完整）
BLOCK_SIZE = 48
BLOCK_MARGIN = MAX_SHIFT
# 每条边选取的背景块数量区间
BLOCKS_PER_AXIS_MIN = 8
BLOCKS_PER_AXIS_MAX = 12

# 背景块至少要在参考帧中具备的纹理强度（按 raw/65535 的灰度比例衡量）
TEXTURE_STD = 0.004
TEXTURE_GRAD = 0.001

# 某个位移下，两块贴片共同有效像素的最低占比
MIN_VALID_CELL_FRACTION = 0.25
# 参与表决的纹理块下限、胜出位移所需票数及相对第二名的倍数
MIN_TEXTURED_BLOCKS = 6
MIN_WIN_VOTES = 4
WIN_VOTE_RATIO = 2.0
# 位移（每轴绝对差）达到该值视为“远处位移”，其残差必须明显比胜出位移差
DISTANT_SHIFT = 5
DISTANT_SCORE_RATIO = 1.15

# 16 位观测的端点：任一通道等于二者之一即为无效（剪切/饱和）
RAW_MIN = 0
RAW_MAX = 65535
RAW_SCALE = 65535.0

# 输入帧数范围
MIN_FRAMES = 3
MAX_FRAMES = 9
