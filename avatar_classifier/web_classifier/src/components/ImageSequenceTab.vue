<script setup lang="ts">
// Tab 区域组件：公共组件 ImageGrid 的包装
// 图片序列由属性传入；行列配置从 pinia 读取（由 Right 功能区表单写入）并透传给公共组件
import ImageGrid from './ImageGrid.vue'
import { useImageSequenceStore } from '../stores/imageSequenceStore'

const props = defineProps<{
  // 功能 id（Tab 插槽名），用于定位各自的网格配置
  tabId: string
  // 图片序列（路径数组）
  images: string[]
}>()

const store = useImageSequenceStore()

// 响应式配置对象：Right 功能区表单修改后，此处自动更新
const grid = store.gridOf(props.tabId)
</script>

<template>
  <ImageGrid :images="images" :n="grid.rows" :m="grid.cols" />
</template>
