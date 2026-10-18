<script setup lang="ts">
import {computed} from 'vue'
import {NTooltip, NEmpty, NTag} from 'naive-ui'
import {useStore} from '../store/useStore'
import type {Servant} from '../types'

const store = useStore()

const statusText = computed(() => {
  if (store.state.queueTotal === 0) return ''
  if (store.isQueueDone()) return '统计完毕'
  return `标记统计中 (${store.state.queueDone}/${store.state.queueTotal})`
})

const isDone = computed(() => store.isQueueDone())

const processingList = computed(() => {
  const entries = Object.entries(store.state.processing)
  if (entries.length === 0) return []
  return entries.map(([idStr, step]) => {
    const id = Number(idStr)
    const servant = store.state.servants.find((s: Servant) => s.id === id)
    const name = servant?.name_cn ?? `#${id}`
    const stepLabel: Record<string, string> = {
      fetching_images: '获取立绘',
      fetching_avatars: '获取标记',
    }
    return {name, step: stepLabel[step] ?? step}
  })
})
</script>

<template>
  <div class="status-bar">
    <span class="status-label">状态：</span>
    <NTooltip v-if="statusText" trigger="hover" :show-arrow="false">
      <template #trigger>
        <NTag :type="isDone ? 'success' : 'info'" size="small" :bordered="false">
          {{ statusText }}
        </NTag>
      </template>
      <div class="tooltip-content">
        <template v-if="processingList.length > 0">
          <div v-for="(item, i) in processingList" :key="i" class="tooltip-item">
            <span class="tooltip-name">{{ item.name }}</span>
            <span class="tooltip-step">{{ item.step }}</span>
          </div>
        </template>
        <NEmpty v-else description="暂无执行中的任务" size="small" />
      </div>
    </NTooltip>
  </div>
</template>

<style scoped>
.status-bar {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 20px;
  background: #fafafa;
  border-top: 1px solid #e8e8e8;
  font-size: 13px;
  min-height: 32px;
}

.status-label {
  color: #666;
}

.tooltip-content {
  min-width: 160px;
  max-height: 200px;
  overflow-y: auto;
}

.tooltip-item {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  padding: 2px 0;
  font-size: 12px;
}

.tooltip-name {
  color: #333;
}

.tooltip-step {
  color: #999;
}
</style>
