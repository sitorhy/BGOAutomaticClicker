<script setup lang="ts">
import {computed} from 'vue'
import {NButton, NInputNumber, NSpace} from 'naive-ui'
import {useStore} from '../store/useStore'
import type {BBoxRange} from '../types'

const store = useStore()

const range = computed(() => store.state.range)
const selectedImage = computed(() => store.state.selectedImage)

const width = computed({
  get: () => range.value ? Math.round(range.value.right - range.value.left) : 0,
  set: (val) => {
    if (!range.value) return
    store.updateRange({...range.value, right: range.value.left + val})
  },
})

const height = computed({
  get: () => range.value ? Math.round(range.value.bottom - range.value.top) : 0,
  set: (val) => {
    if (!range.value) return
    store.updateRange({...range.value, bottom: range.value.top + val})
  },
})

function onFieldUpdate(field: keyof BBoxRange, val: number | null) {
  if (!range.value || val == null) return
  store.updateRange({...range.value, [field]: val})
}

function normalize() {
  if (!range.value) return
  const centerX = (range.value.left + range.value.right) / 2
  const centerY = (range.value.top + range.value.bottom) / 2
  store.updateRange({
    left: Math.round(centerX - 82.5),
    top: Math.round(centerY - 90),
    right: Math.round(centerX + 82.5),
    bottom: Math.round(centerY + 90),
  })
}
</script>

<template>
  <div class="range-display">
    <span class="range-label">头像范围：</span>
    <template v-if="range">
      <NSpace align="center" :size="12" wrap>
        <span class="field-group">
          <span class="field-label">L</span>
          <NInputNumber :value="range.left" :min="0" :precision="0" size="small" style="width: 90px" @update:value="(v: number | null) => onFieldUpdate('left', v)"/>
        </span>
        <span class="field-group">
          <span class="field-label">T</span>
          <NInputNumber :value="range.top" :min="0" :precision="0" size="small" style="width: 90px" @update:value="(v: number | null) => onFieldUpdate('top', v)"/>
        </span>
        <span class="field-group">
          <span class="field-label">R</span>
          <NInputNumber :value="range.right" :min="0" :precision="0" size="small" style="width: 90px" @update:value="(v: number | null) => onFieldUpdate('right', v)"/>
        </span>
        <span class="field-group">
          <span class="field-label">B</span>
          <NInputNumber :value="range.bottom" :min="0" :precision="0" size="small" style="width: 90px" @update:value="(v: number | null) => onFieldUpdate('bottom', v)"/>
        </span>
        <span class="field-group">
          <span class="field-label">宽</span>
          <NInputNumber v-model:value="width" :min="0" :max="512" :precision="0" size="small" style="width: 90px"/>
        </span>
        <span class="field-group">
          <span class="field-label">高</span>
          <NInputNumber v-model:value="height" :min="0" :max="512" :precision="0" size="small" style="width: 90px"/>
        </span>
        <NButton size="small" @click="normalize">规格化</NButton>
      </NSpace>
    </template>
    <span v-else class="range-null">无</span>
    <span v-if="selectedImage" class="range-image-name">— {{ selectedImage }}</span>
  </div>
</template>

<style scoped>
.range-display {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 20px;
  background: #fff;
  border-top: 1px solid #e0e0e0;
  font-size: 13px;
}

.range-label {
  color: #333;
  font-weight: 500;
  flex-shrink: 0;
}

.range-null {
  color: #ff4d4f;
}

.range-image-name {
  color: #999;
  font-size: 12px;
  margin-left: 8px;
  flex-shrink: 0;
}

.field-group {
  display: inline-flex;
  align-items: center;
  gap: 4px;
}

.field-label {
  font-size: 12px;
  color: #666;
  font-weight: 500;
}
</style>
