<script setup lang="ts">
// 功能区子组件：前景合成（步进 / 裁剪掩码图 / 前景图穿梭框 / 合成预览 / 生成序列）
// 挂在「头像定位」「状态裁剪」功能区锁定按钮下方，各自传入本功能的裁剪范围与锁定图
// 合成预览以「锁定目标图片」为底图：按步进矩形裁剪锁定图区域，与前景序列左上角对齐叠加
// 本阶段只做数据采集与静态预览：「生成序列」按钮把组件搜集到的数据打印到控制台
// 穿梭框自绘（Element Plus el-transfer 不支持调整已选列表顺序）：左列点击添加，右列 ↑↓ 调序、✕ 移除
import {computed, onMounted, ref, watch} from 'vue'
import {ElMessage} from 'element-plus'
import type {Rect} from './RectLocateTab.vue'
import type {AssetItem} from '../stores/assetStore'
import {useAssetStore} from '../stores/assetStore'
import {makeOptionFilter} from '../utils/optionFilter'

const props = defineProps<{
  // 裁剪掩码图候选（与头像定位的掩码图下拉同源，由父组件传入）
  maskOptions: AssetItem[]
  // 状态裁剪的最大 / 最小范围（步进矩形插值端点，生成序列时随搜集数据一并打印）
  maxRect: Rect
  minRect: Rect
  // 锁定的目标图片（合成预览底图），未锁定时为 null
  target: { path: string; url: string } | null
}>()

const assetStore = useAssetStore()

// —— 步进：滑块 [1, 30]，默认 10 ——
const step = ref(10)

// —— 裁剪掩码图：下拉抄头像裁剪功能的掩码数据源（props.maskOptions），过滤逻辑同样复用 ——
const maskPath = ref('')
const maskFilter = makeOptionFilter(() => props.maskOptions)
const maskItem = computed(() => props.maskOptions.find(a => a.path === maskPath.value) ?? null)

// —— 参照图：下拉数据源与 HomeView「头像路径」一致（assetStore.assets），显示在预览区域右方 ——
const referencePath = ref('')
const referenceFilter = makeOptionFilter(() => assetStore.assets)
const referenceItem = computed(() =>
    assetStore.assets.find(a => a.path === referencePath.value) ?? null
)

// —— 前景图：数据源为 assetStore.fetchForegroundAssets（foreground 目录）——
const foregroundAssets = ref<AssetItem[]>([])

onMounted(async () => {
  try {
    foregroundAssets.value = await assetStore.fetchForegroundAssets()
  } catch (e) {
    console.error('Failed to fetch foreground assets:', e)
  }
})

// 已选前景图的 path 数组，顺序即合成叠放顺序
const selectedPaths = ref<string[]>([])

const availableItems = computed(() =>
    foregroundAssets.value.filter(a => !selectedPaths.value.includes(a.path))
)

const selectedItems = computed(() =>
    selectedPaths.value
        .map(p => foregroundAssets.value.find(a => a.path === p))
        .filter((a): a is AssetItem => a !== undefined)
)

function addForeground(item: AssetItem) {
  selectedPaths.value.push(item.path)
}

function removeForeground(index: number) {
  selectedPaths.value.splice(index, 1)
}

// 与相邻项交换位置，实现叠放顺序调整
function moveForeground(index: number, offset: -1 | 1) {
  const target = index + offset
  if (target < 0 || target >= selectedPaths.value.length) return
  const paths = [...selectedPaths.value]
  ;[paths[index], paths[target]] = [paths[target], paths[index]]
  selectedPaths.value = paths
}

// —— 步进矩形：最大矩形 → 最小矩形，坐标与尺寸线性插值 ——
// 步进滑块 t = 等距取样个数，第 i 个矩形取 k = i/t（不含最大端、含最小端，t=1 即最小矩形）；
// 预览序号滑块 n ∈ [1, t] 选择当前用作底图裁剪的是第几个步进矩形
const stepIndex = ref(1)

// 步进个数变小时，序号钳制回有效范围
watch(step, t => {
  if (stepIndex.value > t) stepIndex.value = t
})

const stepRect = computed<Rect>(() => {
  const k = stepIndex.value / step.value
  const lerp = (a: number, b: number) => Math.round(a + (b - a) * k)
  return {
    left: lerp(props.maxRect.left, props.minRect.left),
    top: lerp(props.maxRect.top, props.minRect.top),
    right: lerp(props.maxRect.right, props.minRect.right),
    bottom: lerp(props.maxRect.bottom, props.minRect.bottom),
  }
})

// —— 合成预览：点击「预览生成」检测数据完备后，按掩码图尺寸渲染叠加区域 ——
const previewSize = ref<{ w: number; h: number } | null>(null)
// 预览使用生成时刻的前景图快照，之后调整穿梭框不影响已生成的预览
const previewFore = ref<AssetItem[]>([])
// 底图快照：生成时刻的锁定图
const previewTarget = ref<{ path: string; url: string } | null>(null)

function loadImageSize(url: string): Promise<{ w: number; h: number }> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve({w: img.naturalWidth, h: img.naturalHeight})
    img.onerror = () => reject(new Error('图片加载失败'))
    img.src = url
  })
}

async function generatePreview() {
  // 完备性检测：锁定图（底图）、掩码图（决定画布尺寸）与前景图（至少一张）缺一不可
  if (!props.target) {
    ElMessage.warning('请先锁定目标图片')
    return
  }
  if (!maskItem.value) {
    ElMessage.warning('请选择裁剪掩码图')
    return
  }
  if (selectedItems.value.length === 0) {
    ElMessage.warning('请选择前景图')
    return
  }
  try {
    previewSize.value = await loadImageSize(maskItem.value.url)
    previewFore.value = [...selectedItems.value]
    previewTarget.value = {...props.target}
  } catch (e) {
    ElMessage.error(`生成预览失败：${e instanceof Error ? e.message : String(e)}`)
  }
}

// —— 生成序列：本阶段仅打印组件搜集到的数据 ——
function generateSequence() {
  const data = {
    step: step.value,
    stepIndex: stepIndex.value,
    stepRect: stepRect.value,
    target: props.target ? {path: props.target.path} : null,
    reference: referenceItem.value ? {file: referenceItem.value.file, path: referenceItem.value.path} : null,
    mask: maskItem.value ? {file: maskItem.value.file, path: maskItem.value.path} : null,
    foregrounds: selectedItems.value.map(a => ({file: a.file, path: a.path})),
    maxRect: props.maxRect,
    minRect: props.minRect,
  }
  console.log('[前景合成] 搜集数据:', JSON.stringify(data, null, 2))
  ElMessage.success('已打印搜集数据到控制台')
}
</script>

<template>
  <div class="foreground-compose-panel">
    <el-form label-position="top" size="default">
      <el-form-item label="步进">
        <el-slider
            v-model="step"
            :min="1"
            :max="30"
            show-input
            input-size="small"
            class="step-slider"
        />
      </el-form-item>

      <el-form-item label="裁剪掩码图">
        <el-select
            v-model="maskPath"
            placeholder="选择图片"
            clearable
            filterable
            :filter-method="(q: string) => (maskFilter.query.value = q)"
            @visible-change="maskFilter.onDropdownVisible"
            style="width: 100%"
        >
          <el-option
              v-for="item in maskFilter.filtered.value"
              :key="item.path"
              :label="item.file"
              :value="item.path"
          >
            <div class="path-option">
              <img :src="item.url" class="path-option-thumb" alt="" loading="lazy"/>
              <span class="path-option-name">{{ item.file }}</span>
            </div>
          </el-option>
        </el-select>
      </el-form-item>

      <el-form-item label="前景图">
        <div class="fg-transfer">
          <!-- 左列：候选（点击添加到右侧，按添加顺序叠放） -->
          <div class="fg-col">
            <div class="fg-col-title">候选 ({{ availableItems.length }})</div>
            <ul class="fg-list">
              <li
                  v-for="item in availableItems"
                  :key="item.path"
                  class="fg-item"
                  @click="addForeground(item)"
              >
                <img :src="item.url" class="fg-thumb" alt="" loading="lazy"/>
                <span class="fg-name">{{ item.file }}</span>
              </li>
            </ul>
          </div>
          <!-- 右列：已选（↑↓ 调整叠放顺序，✕ 移回候选） -->
          <div class="fg-col">
            <div class="fg-col-title">已选 ({{ selectedItems.length }})</div>
            <ul class="fg-list">
              <li v-for="(item, i) in selectedItems" :key="item.path" class="fg-item">
                <span class="fg-order-btn" :class="{ disabled: i === 0 }" title="上移" @click="moveForeground(i, -1)">↑</span>
                <span class="fg-order-btn" :class="{ disabled: i === selectedItems.length - 1 }" title="下移" @click="moveForeground(i, 1)">↓</span>
                <img :src="item.url" class="fg-thumb" alt="" loading="lazy"/>
                <span class="fg-name">{{ item.file }}</span>
                <span class="fg-order-btn" title="移除" @click="removeForeground(i)">✕</span>
              </li>
            </ul>
          </div>
        </div>
      </el-form-item>
    </el-form>

    <!-- 合成预览：静态展示 锁定图步进矩形裁剪底图 + 前景图按序左上角叠加 -->
    <div class="compose-section">
      <div class="compose-title">合成预览</div>
      <el-button type="primary" @click="generatePreview">预览生成</el-button>
      <!-- 序号滑块：1 <= n <= t，选第 n 个步进矩形作为底图裁剪区域 -->
      <div class="index-row">
        <span class="index-label">序号 n</span>
        <el-slider
            v-model="stepIndex"
            :min="1"
            :max="step"
            show-input
            input-size="small"
            class="step-slider"
        />
      </div>
      <!-- 参照图：数据源抄 HomeView「头像路径」下拉，选中后显示在预览区域右方 -->
      <div class="reference-row">
        <span class="index-label">参照图</span>
        <el-select
            v-model="referencePath"
            placeholder="选择图片"
            clearable
            filterable
            :filter-method="(q: string) => (referenceFilter.query.value = q)"
            @visible-change="referenceFilter.onDropdownVisible"
            style="width: 100%"
        >
          <el-option
              v-for="item in referenceFilter.filtered.value"
              :key="item.path"
              :label="item.file"
              :value="item.path"
          >
            <div class="path-option">
              <img :src="item.url" class="path-option-thumb" alt="" loading="lazy"/>
              <span class="path-option-name">{{ item.file }}</span>
            </div>
          </el-option>
        </el-select>
      </div>
      <div class="preview-body">
        <div v-if="previewSize" class="preview-scroll">
          <!-- 画布尺寸 = 掩码图原始尺寸，马赛克棋盘底衬托透明区域，超出画布的内容裁掉 -->
          <div
              class="compose-stage"
              :style="{ width: `${previewSize.w}px`, height: `${previewSize.h}px` }"
          >
            <!-- 底图：锁定图裁剪当前步进矩形区域，与前景序列左上角对齐 -->
            <div
                v-if="previewTarget"
                class="compose-base"
                :style="{
                  width: `${stepRect.right - stepRect.left}px`,
                  height: `${stepRect.bottom - stepRect.top}px`,
                }"
            >
              <img
                  :src="previewTarget.url"
                  class="compose-base-img"
                  :style="{ left: `${-stepRect.left}px`, top: `${-stepRect.top}px` }"
                  alt=""
              />
            </div>
            <img
                v-for="(item, i) in previewFore"
                :key="item.path"
                :src="item.url"
                class="compose-layer"
                :style="{ zIndex: i + 1 }"
                alt=""
            />
          </div>
        </div>
        <el-empty v-else description="点击「预览生成」查看合成效果" :image-size="48"/>
        <!-- 参照图显示在预览区域右方 -->
        <img v-if="referenceItem" :src="referenceItem.url" class="reference-img" alt="参照图"/>
      </div>
    </div>

    <!-- 生成序列：本阶段打印搜集数据 -->
    <div class="compose-section">
      <el-button type="success" @click="generateSequence">生成序列</el-button>
    </div>
  </div>
</template>

<style scoped>
.foreground-compose-panel {
  width: 100%;
}

/* 滑块 + 数值框在窄侧栏内占满一行 */
.step-slider {
  width: 100%;
}

/* 掩码图下拉选项：缩略图 + 文件名（与 HomeView 的下拉选项样式一致） */
.path-option {
  display: flex;
  align-items: center;
  gap: 8px;
  line-height: 1.2;
}

.path-option-thumb {
  width: 24px;
  height: 24px;
  flex-shrink: 0;
  object-fit: contain;
  border-radius: 2px;
  background-color: #f5f7fa;
}

.path-option-name {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* 自绘穿梭框：左右两列 */
.fg-transfer {
  width: 100%;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 6px;
}

.fg-col {
  border: 1px solid #dcdfe6;
  border-radius: 4px;
  background-color: #fff;
  overflow: hidden;
}

.fg-col-title {
  font-size: 12px;
  color: #909399;
  padding: 3px 6px;
  border-bottom: 1px solid #ebeef5;
  background-color: #f5f7fa;
}

.fg-list {
  list-style: none;
  margin: 0;
  padding: 2px;
  max-height: 160px;
  overflow-y: auto;
}

.fg-item {
  display: flex;
  align-items: center;
  gap: 3px;
  padding: 2px;
  border-radius: 3px;
  font-size: 12px;
  cursor: pointer;
}

.fg-item:hover {
  background-color: #f5f7fa;
}

.fg-thumb {
  width: 20px;
  height: 20px;
  flex-shrink: 0;
  object-fit: contain;
}

.fg-name {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.fg-order-btn {
  flex-shrink: 0;
  width: 16px;
  text-align: center;
  font-size: 12px;
  color: #409eff;
  cursor: pointer;
  user-select: none;
}

.fg-order-btn:hover {
  color: #79bbff;
}

.fg-order-btn.disabled {
  color: #c0c4cc;
  cursor: default;
  pointer-events: none;
}

/* 合成预览区 */
.compose-section {
  margin-top: 8px;
}

.compose-title {
  font-size: 13px;
  font-weight: 600;
  color: #606266;
  margin-bottom: 4px;
}

/* 预览序号滑块行 */
.index-row {
  margin-top: 8px;
}

.index-label {
  font-size: 12px;
  color: #909399;
}

/* 参照图下拉行 */
.reference-row {
  margin-top: 8px;
}

/* 预览区域 + 右方参照图 */
.preview-body {
  display: flex;
  align-items: flex-start;
  gap: 8px;
}

.reference-img {
  flex-shrink: 0;
  max-height: 320px;
  max-width: 180px;
  object-fit: contain;
  border: 1px solid #e0e0e0;
  border-radius: 4px;
  background-color: #fff;
}

/* 掩码图可能大于侧栏宽度，外层出滚动条；无内外边框 */
.preview-scroll {
  margin-top: 8px;
  max-height: 320px;
  overflow: auto;
}

/* 平铺马赛克棋盘底；画布超出掩码尺寸的部分不显示 */
.compose-stage {
  position: relative;
  overflow: hidden;
  box-sizing: border-box;
  background-color: #fff;
  background-image:
      linear-gradient(45deg, #ccc 25%, transparent 25%, transparent 75%, #ccc 75%),
      linear-gradient(45deg, #ccc 25%, transparent 25%, transparent 75%, #ccc 75%);
  background-size: 16px 16px;
  background-position: 0 0, 8px 8px;
}

/* 底图裁剪窗：尺寸 = 步进矩形，左上角对齐画布 */
.compose-base {
  position: absolute;
  top: 0;
  left: 0;
  overflow: hidden;
}

/* 锁定图原始尺寸渲染，反向偏移步进矩形左上角实现裁剪 */
.compose-base-img {
  position: absolute;
  display: block;
}

/* 前景图按顺序叠加，对齐画布左上角 */
.compose-layer {
  position: absolute;
  top: 0;
  left: 0;
  display: block;
}
</style>
