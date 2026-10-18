<script setup lang="ts">
import {computed, ref} from 'vue'
import {useStore} from '../store/useStore'
import type {BBoxRange} from '../types'
import BoundingBox from './BoundingBox.vue'

const store = useStore()
const selectedServantId = computed(() => store.state.selectedServantId)
const selectedImage = computed(() => store.state.selectedImage)
const range = computed(() => store.state.range)

const naturalWidth = ref(0)
const naturalHeight = ref(0)

const imageUrl = computed(() => {
  if (selectedServantId.value == null || !selectedImage.value) return ''
  return store.getImageUrl(selectedServantId.value, selectedImage.value)
})

function onRangeUpdate(r: BBoxRange) {
  store.updateRange(r)
}

function onImageLoad(e: Event) {
  const img = e.target as HTMLImageElement
  naturalWidth.value = img.naturalWidth
  naturalHeight.value = img.naturalHeight
}

function handleKeydown(e: KeyboardEvent) {
  if (!range.value) return
  const step = e.shiftKey ? 5 : 1
  const r = {...range.value}

  switch (e.key) {
    case 'ArrowLeft':
      e.preventDefault()
      r.left = Math.max(0, r.left - step)
      r.right = Math.max(0, r.right - step)
      break
    case 'ArrowRight':
      e.preventDefault()
      r.left = r.left + step
      r.right = r.right + step
      break
    case 'ArrowUp':
      e.preventDefault()
      r.top = Math.max(0, r.top - step)
      r.bottom = Math.max(0, r.bottom - step)
      break
    case 'ArrowDown':
      e.preventDefault()
      r.top = r.top + step
      r.bottom = r.bottom + step
      break
    default:
      return
  }
  store.updateRange(r)
}
</script>

<template>
  <div class="image-canvas" @keydown="handleKeydown" tabindex="0">
    <div v-if="!imageUrl" class="canvas-placeholder">
      <div class="placeholder-text">请从左侧选择从者，然后选择图片</div>
    </div>
    <div v-else class="canvas-content">
      <img
          :src="imageUrl"
          class="canvas-image"
          @load="onImageLoad"
          draggable="false"
      />
      <BoundingBox
          v-if="range && naturalWidth && naturalHeight"
          :range="range"
          :natural-width="naturalWidth"
          :natural-height="naturalHeight"
          @update:range="onRangeUpdate"
      />
    </div>
  </div>
</template>

<style scoped>
.image-canvas {
  flex: 1;
  background: #f0f0f0;
  overflow: auto;
  position: relative;
  outline: none;
}

.image-canvas:focus {
  box-shadow: inset 0 0 0 2px #1677ff;
}

.canvas-placeholder {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 100%;
}

.placeholder-text {
  color: #bbb;
  font-size: 16px;
}

.canvas-content {
  position: relative;
  display: inline-block;
  line-height: 0;
}

.canvas-image {
  display: block;
  user-select: none;
  -webkit-user-drag: none;
}
</style>
