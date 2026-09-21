<script setup lang="ts">
// 公共功能区组件（放 Right 操作区）：编辑最大 / 最小矩形范围
// 与 RectLocateTab 配对，供「头像定位」「状态裁剪」等功能复用
// 数值修改通过 v-model:max-rect / v-model:min-rect 回传，与 Tab 区域的拖动编辑保持同步
// 图片路径、掩码图等表单下拉由使用方在 #form 插槽中提供（见 HomeView）
import type { Rect } from './RectLocateTab.vue'

const props = defineProps<{
  // 最大范围矩形
  maxRect: Rect
  // 最小范围矩形
  minRect: Rect
}>()

const emit = defineEmits<{
  (e: 'update:maxRect', rect: Rect): void
  (e: 'update:minRect', rect: Rect): void
}>()

const fields: { key: keyof Rect; label: string }[] = [
  { key: 'left', label: 'left' },
  { key: 'top', label: 'top' },
  { key: 'right', label: 'right' },
  { key: 'bottom', label: 'bottom' },
]

// 编辑单个坐标值：保持 right > left / bottom > top 后回传
function updateField(rectKey: 'maxRect' | 'minRect', field: keyof Rect, value: number | undefined) {
  const next: Rect = { ...props[rectKey], [field]: Math.max(0, Math.round(value ?? 0)) }
  next.right = Math.max(next.right, next.left + 1)
  next.bottom = Math.max(next.bottom, next.top + 1)
  if (rectKey === 'maxRect') {
    emit('update:maxRect', next)
  } else {
    emit('update:minRect', next)
  }
}
</script>

<template>
  <div class="rect-locate-panel">
    <slot></slot>
    <el-form label-position="top" size="default" class="locate-form">
      <slot name="form"></slot>

      <el-form-item label="最大范围">
        <div class="rect-editor editor-max">
          <label v-for="f in fields" :key="f.key" class="rect-field">
            <span class="rect-field-label">{{ f.label }}</span>
            <el-input-number
              :model-value="maxRect[f.key]"
              :min="0"
              :controls="false"
              size="small"
              class="rect-input"
              @update:model-value="updateField('maxRect', f.key, $event)"
            />
          </label>
        </div>
      </el-form-item>

      <el-form-item label="最小范围">
        <div class="rect-editor editor-min">
          <label v-for="f in fields" :key="f.key" class="rect-field">
            <span class="rect-field-label">{{ f.label }}</span>
            <el-input-number
              :model-value="minRect[f.key]"
              :min="0"
              :controls="false"
              size="small"
              class="rect-input"
              @update:model-value="updateField('minRect', f.key, $event)"
            />
          </label>
        </div>
      </el-form-item>
    </el-form>
  </div>
</template>

<style scoped>
.rect-locate-panel {
  width: 100%;
}

.locate-form :deep(.el-form-item__label) {
  padding-bottom: 2px;
  font-weight: 600;
}

/* 2 x 2 排布的四个坐标输入框 */
.rect-editor {
  width: 100%;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 6px;
  padding: 6px;
  box-sizing: border-box;
  border-radius: 4px;
  background-color: #fff;
}

.editor-max {
  border: 1px solid red;
}

.editor-min {
  border: 1px solid green;
}

.rect-field {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.rect-field-label {
  font-size: 12px;
  color: #909399;
}

.rect-input {
  width: 100%;
}

.rect-input :deep(.el-input__inner) {
  text-align: left;
}
</style>
