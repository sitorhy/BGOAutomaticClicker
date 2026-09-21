<script setup lang="ts">
// 公共组件（矩形范围定位 / 编辑核心）：显示图片并在上方标记最大 / 最小矩形范围
// 供「头像定位」「状态裁剪」等功能复用
// 范围编辑：拖动矩形本体移动位置，拖动 8 个缩放手柄改变 (left, top, right, bottom)
// 修改结果通过 v-model:max-rect / v-model:min-rect 回传父组件
import { ref } from 'vue'

export interface Rect {
  left: number
  top: number
  right: number
  bottom: number
}

type RectKey = 'max' | 'min'
type DragMode = 'move' | 'n' | 's' | 'w' | 'e' | 'nw' | 'ne' | 'sw' | 'se'

const props = withDefaults(
  defineProps<{
    // 选中的图片路径（由父组件从 Right 功能区下拉选择传入）
    imagePath?: string | null
    // 最大范围矩形（按图片无缩放渲染尺寸定义）
    maxRect: Rect
    // 最小范围矩形（按图片无缩放渲染尺寸定义）
    minRect: Rect
    // 最大矩形边框颜色 / 宽度
    maxColor?: string
    maxBorderWidth?: number
    // 最小矩形边框颜色 / 宽度
    minColor?: string
    minBorderWidth?: number
  }>(),
  {
    imagePath: null,
    maxColor: 'red',
    maxBorderWidth: 2,
    minColor: 'green',
    minBorderWidth: 2,
  }
)

const emit = defineEmits<{
  (e: 'update:maxRect', rect: Rect): void
  (e: 'update:minRect', rect: Rect): void
}>()

// 缩放手柄（8 向）
const handles: DragMode[] = ['nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w']

// 图片原始尺寸：矩形不允许拖出图片边界
const imageSize = ref<{ w: number; h: number } | null>(null)

function onImageLoad(e: Event) {
  const img = e.target as HTMLImageElement
  imageSize.value = { w: img.naturalWidth, h: img.naturalHeight }
}

// 依据矩形计算绝对定位样式（图片不缩放，坐标即像素坐标）
function rectStyle(rect: Rect, color: string, borderWidth: number) {
  return {
    left: `${rect.left}px`,
    top: `${rect.top}px`,
    width: `${rect.right - rect.left}px`,
    height: `${rect.bottom - rect.top}px`,
    borderColor: color,
    borderWidth: `${borderWidth}px`,
  }
}

function currentRect(key: RectKey): Rect {
  return key === 'max' ? props.maxRect : props.minRect
}

// 限制矩形在图片范围内，且保持最小尺寸
function clampRect(r: Rect): Rect {
  const maxW = imageSize.value?.w ?? Number.MAX_SAFE_INTEGER
  const maxH = imageSize.value?.h ?? Number.MAX_SAFE_INTEGER
  let { left, top, right, bottom } = r
  left = Math.max(0, Math.min(left, maxW - 1))
  top = Math.max(0, Math.min(top, maxH - 1))
  right = Math.max(left + 1, Math.min(right, maxW))
  bottom = Math.max(top + 1, Math.min(bottom, maxH))
  return {
    left: Math.round(left),
    top: Math.round(top),
    right: Math.round(right),
    bottom: Math.round(bottom),
  }
}

// 按拖拽模式与位移量计算新矩形
function applyDrag(orig: Rect, mode: DragMode, dx: number, dy: number): Rect {
  let { left, top, right, bottom } = orig
  if (mode === 'move' || mode === 'w' || mode === 'nw' || mode === 'sw') left += dx
  if (mode === 'move' || mode === 'e' || mode === 'ne' || mode === 'se') right += dx
  if (mode === 'move' || mode === 'n' || mode === 'nw' || mode === 'ne') top += dy
  if (mode === 'move' || mode === 's' || mode === 'sw' || mode === 'se') bottom += dy
  return clampRect({ left, top, right, bottom })
}

interface DragState {
  key: RectKey
  mode: DragMode
  startX: number
  startY: number
  orig: Rect
}
let drag: DragState | null = null

function onPointerDown(e: PointerEvent, key: RectKey, mode: DragMode) {
  drag = { key, mode, startX: e.clientX, startY: e.clientY, orig: { ...currentRect(key) } }
  window.addEventListener('pointermove', onPointerMove)
  window.addEventListener('pointerup', onPointerUp)
  e.preventDefault()
  e.stopPropagation()
}

function onPointerMove(e: PointerEvent) {
  if (!drag) return
  const rect = applyDrag(drag.orig, drag.mode, e.clientX - drag.startX, e.clientY - drag.startY)
  if (drag.key === 'max') {
    emit('update:maxRect', rect)
  } else {
    emit('update:minRect', rect)
  }
}

function onPointerUp() {
  drag = null
  window.removeEventListener('pointermove', onPointerMove)
  window.removeEventListener('pointerup', onPointerUp)
}
</script>

<template>
  <div class="rect-locate-tab">
    <!-- 允许滚动：矩形按图片无缩放尺寸定义，可能超出可视范围 -->
    <div v-if="imagePath" class="locate-scroll">
      <div class="locate-stage">
        <img :src="imagePath" class="locate-image" alt="locate" @load="onImageLoad" />

        <!-- 最大矩形标记：本体拖动移动，手柄缩放 -->
        <div
          class="locate-rect locate-rect-editable"
          :style="rectStyle(props.maxRect, props.maxColor, props.maxBorderWidth)"
          @pointerdown="onPointerDown($event, 'max', 'move')"
        >
          <span class="locate-rect-label" :style="{ color: props.maxColor }">最大</span>
          <span
            v-for="h in handles"
            :key="h"
            class="rect-handle"
            :class="`handle-${h}`"
            :style="{ borderColor: props.maxColor }"
            @pointerdown.stop="onPointerDown($event, 'max', h)"
          ></span>
        </div>

        <!-- 最小矩形标记 -->
        <div
          class="locate-rect locate-rect-editable"
          :style="rectStyle(props.minRect, props.minColor, props.minBorderWidth)"
          @pointerdown="onPointerDown($event, 'min', 'move')"
        >
          <span class="locate-rect-label" :style="{ color: props.minColor }">最小</span>
          <span
            v-for="h in handles"
            :key="h"
            class="rect-handle"
            :class="`handle-${h}`"
            :style="{ borderColor: props.minColor }"
            @pointerdown.stop="onPointerDown($event, 'min', h)"
          ></span>
        </div>
      </div>
    </div>

    <el-empty v-else description="请选择图片" />
  </div>
</template>

<style scoped>
.rect-locate-tab {
  height: 100%;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.locate-scroll {
  flex: 1;
  overflow: auto;
}

.locate-stage {
  position: relative;
  display: inline-block;
  line-height: 0;
}

.locate-image {
  display: block;
  /* 不缩放：以图片原始尺寸渲染 */
  max-width: none;
}

.locate-rect {
  position: absolute;
  box-sizing: border-box;
  border-style: solid;
  pointer-events: none;
}

.locate-rect-editable {
  pointer-events: auto;
  cursor: move;
}

.rect-handle {
  position: absolute;
  width: 10px;
  height: 10px;
  box-sizing: border-box;
  background-color: #fff;
  border-style: solid;
  border-width: 2px;
  border-radius: 2px;
}

.handle-nw { left: -5px; top: -5px; cursor: nwse-resize; }
.handle-n { left: calc(50% - 5px); top: -5px; cursor: ns-resize; }
.handle-ne { right: -5px; top: -5px; cursor: nesw-resize; }
.handle-e { right: -5px; top: calc(50% - 5px); cursor: ew-resize; }
.handle-se { right: -5px; bottom: -5px; cursor: nwse-resize; }
.handle-s { left: calc(50% - 5px); bottom: -5px; cursor: ns-resize; }
.handle-sw { left: -5px; bottom: -5px; cursor: nesw-resize; }
.handle-w { left: -5px; top: calc(50% - 5px); cursor: ew-resize; }

.locate-rect-label {
  position: absolute;
  top: 2px;
  left: 2px;
  font-size: 12px;
  line-height: 1.2;
  background-color: rgba(255, 255, 255, 0.7);
  padding: 0 4px;
  border-radius: 2px;
  pointer-events: none;
}
</style>
