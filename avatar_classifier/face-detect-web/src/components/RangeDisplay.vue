<script setup lang="ts">
import {computed} from 'vue'
import {useStore} from '../store/useStore'

const store = useStore()

const range = computed(() => store.state.range)
const selectedImage = computed(() => store.state.selectedImage)

const rangeText = computed(() => {
  if (!range.value) return 'null'
  const r = range.value
  return `(${Math.round(r.left)}, ${Math.round(r.top)}, ${Math.round(r.right)}, ${Math.round(r.bottom)})`
})
</script>

<template>
  <div class="range-display">
    <span class="range-label">头像范围：</span>
    <span v-if="range" class="range-value">{{ rangeText }}</span>
    <span v-else class="range-null">无</span>
    <span v-if="selectedImage" class="range-image-name">— {{ selectedImage }}</span>
  </div>
</template>

<style scoped>
.range-display {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 20px;
  background: #fff;
  border-top: 1px solid #e0e0e0;
  font-size: 14px;
  font-family: 'Consolas', 'Monaco', monospace;
}

.range-label {
  color: #333;
  font-weight: 500;
}

.range-value {
  color: #1677ff;
}

.range-null {
  color: #ff4d4f;
}

.range-image-name {
  color: #999;
  font-size: 12px;
  margin-left: 8px;
}
</style>
