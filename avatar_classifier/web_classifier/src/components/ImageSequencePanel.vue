<script setup lang="ts">
// 功能区组件（放 Right 操作区）：图片序列网格的行 / 列表单
// 选择结果写入 pinia，Tab 区域组件（同一 tabId）读取后驱动公共组件渲染
import { useImageSequenceStore } from '../stores/imageSequenceStore'

const props = defineProps<{
  // 功能 id（Tab 插槽名），每个功能各自一份配置
  tabId: string
}>()

const store = useImageSequenceStore()

// 行 / 列可选范围：[1, 10]
const rangeOptions = Array.from({ length: 10 }, (_, i) => i + 1)

// 响应式配置对象：表单直接双向绑定到 store
const grid = store.gridOf(props.tabId)
</script>

<template>
  <div class="image-sequence-panel">
    <el-form label-position="top" class="sequence-form">
      <el-form-item label="行">
        <el-select v-model="grid.rows" style="width: 100%">
          <el-option v-for="num in rangeOptions" :key="num" :label="num" :value="num" />
        </el-select>
      </el-form-item>

      <el-form-item label="列">
        <el-select v-model="grid.cols" style="width: 100%">
          <el-option v-for="num in rangeOptions" :key="num" :label="num" :value="num" />
        </el-select>
      </el-form-item>
    </el-form>
  </div>
</template>

<style scoped>
.image-sequence-panel {
  width: 100%;
}

.sequence-form :deep(.el-form-item__label) {
  padding-bottom: 2px;
  font-weight: 600;
}
</style>
