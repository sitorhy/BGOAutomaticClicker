<script setup lang="ts">
import { ref, watch } from 'vue'

export interface TabConfig {
  label: string
  slot: string
}

const props = defineProps<{
  tabs: TabConfig[]
}>()

const emit = defineEmits<{
  (e: 'change', slot: string): void
}>()

// 默认激活第一个 Tab
const active = ref(props.tabs[0]?.slot ?? '')

watch(active, (slot) => {
  if (slot) emit('change', slot)
})

// 配置变化时，若当前激活项已不存在则回退到第一项
watch(
  () => props.tabs,
  (tabs) => {
    if (!tabs.some(t => t.slot === active.value)) {
      active.value = tabs[0]?.slot ?? ''
    }
  }
)
</script>

<template>
  <el-tabs v-model="active" class="tabs-panel-component">
    <el-tab-pane
      v-for="tab in tabs"
      :key="tab.slot"
      :label="tab.label"
      :name="tab.slot"
    >
      <!-- 标签内容 = 父组件提供的同名插槽；插槽不存在则不显示内容 -->
      <slot v-if="$slots[tab.slot]" :name="tab.slot" />
    </el-tab-pane>
  </el-tabs>
</template>

<style scoped>
.tabs-panel-component {
  height: 100%;
  display: flex;
  flex-direction: column;
}

.tabs-panel-component :deep(.el-tabs__content) {
  flex: 1;
  overflow: auto;
}

/* 面板撑满内容区高度，使需要滚动的组件（如大图定位）在自身内部滚动 */
.tabs-panel-component :deep(.el-tab-pane) {
  height: 100%;
}
</style>
