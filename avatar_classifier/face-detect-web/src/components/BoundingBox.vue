<script setup lang="ts">
import {ref, watch} from 'vue'
import type {BBoxRange} from '../types'

const props = defineProps<{
  range: BBoxRange
  naturalWidth: number
  naturalHeight: number
}>()

const emit = defineEmits<{
  (e: 'update:range', range: BBoxRange): void
}>()

const box = ref({...props.range})

watch(
    () => props.range,
    (val) => {
      box.value = {...val}
    },
    {deep: true},
)

type Handle = 'nw' | 'n' | 'ne' | 'e' | 'se' | 's' | 'sw' | 'w' | 'body'

const dragging = ref(false)
const activeHandle = ref<Handle | null>(null)
const containerRect = ref<DOMRect | null>(null)
const startBox = ref({left: 0, top: 0, right: 0, bottom: 0})

function toPctX(absX: number): number {
  return (absX / props.naturalWidth) * 100
}

function toPctY(absY: number): number {
  return (absY / props.naturalHeight) * 100
}

function toAbsX(pctX: number): number {
  return (pctX / 100) * props.naturalWidth
}

function toAbsY(pctY: number): number {
  return (pctY / 100) * props.naturalHeight
}

function getHandleClass(h: Handle) {
  if (h === 'body') return ''
  const map: Record<string, string> = {
    nw: 'handle-nw', n: 'handle-n', ne: 'handle-ne',
    e: 'handle-e', se: 'handle-se', s: 'handle-s',
    sw: 'handle-sw', w: 'handle-w',
  }
  return map[h] ?? ''
}

function getCursor(h: Handle) {
  if (h === 'body') return 'move'
  const map: Record<string, string> = {
    nw: 'nw-resize', n: 'n-resize', ne: 'ne-resize',
    e: 'e-resize', se: 'se-resize', s: 's-resize',
    sw: 'sw-resize', w: 'w-resize',
  }
  return map[h] ?? 'default'
}

function clamp(v: number, min: number, max: number) {
  return Math.max(min, Math.min(max, v))
}

function onHandleDown(e: MouseEvent, handle: Handle) {
  e.stopPropagation()
  e.preventDefault()
  const el = (e.currentTarget as HTMLElement).closest('.bbox-container') as HTMLElement
  containerRect.value = el.getBoundingClientRect()
  startBox.value = {...box.value}
  activeHandle.value = handle
  dragging.value = true
  document.addEventListener('mousemove', onMove)
  document.addEventListener('mouseup', onUp)
}

function onBodyDown(e: MouseEvent) {
  if (dragging.value) return
  e.preventDefault()
  const el = (e.currentTarget as HTMLElement).closest('.bbox-container') as HTMLElement
  containerRect.value = el.getBoundingClientRect()
  startBox.value = {...box.value}
  activeHandle.value = 'body'
  dragging.value = true
  const rect = containerRect.value
  const startXPct = ((e.clientX - rect.left) / rect.width) * 100
  const startYPct = ((e.clientY - rect.top) / rect.height) * 100
  ;(onMove as any).__offsetX = startXPct - toPctX(box.value.left)
  ;(onMove as any).__offsetY = startYPct - toPctY(box.value.top)
  document.addEventListener('mousemove', onMove)
  document.addEventListener('mouseup', onUp)
}

function onMove(e: MouseEvent) {
  if (!dragging.value || !containerRect.value || !activeHandle.value) return
  if (!props.naturalWidth || !props.naturalHeight) return
  const rect = containerRect.value
  const xPct = ((e.clientX - rect.left) / rect.width) * 100
  const yPct = ((e.clientY - rect.top) / rect.height) * 100
  const b = startBox.value
  const MIN_ABS = 5
  const minPctX = toPctX(MIN_ABS)
  const minPctY = toPctY(MIN_ABS)

  let newLeft = b.left
  let newTop = b.top
  let newRight = b.right
  let newBottom = b.bottom

  const h = activeHandle.value

  if (h === 'body') {
    const ox = (onMove as any).__offsetX as number
    const oy = (onMove as any).__offsetY as number
    const newLeftPct = clamp(xPct - ox, 0, 100 - minPctX)
    const newTopPct = clamp(yPct - oy, 0, 100 - minPctY)
    const widthPct = toPctX(b.right - b.left)
    const heightPct = toPctY(b.bottom - b.top)
    newLeft = Math.round(toAbsX(newLeftPct))
    newTop = Math.round(toAbsY(newTopPct))
    newRight = Math.round(toAbsX(newLeftPct + widthPct))
    newBottom = Math.round(toAbsY(newTopPct + heightPct))
  } else {
    if (h.includes('w')) {
      newLeft = Math.round(toAbsX(clamp(xPct, 0, toPctX(b.right) - minPctX)))
    }
    if (h.includes('e')) {
      newRight = Math.round(toAbsX(clamp(xPct, toPctX(b.left) + minPctX, 100)))
    }
    if (h === 'nw' || h === 'n' || h === 'ne') {
      newTop = Math.round(toAbsY(clamp(yPct, 0, toPctY(b.bottom) - minPctY)))
    }
    if (h === 'sw' || h === 's' || h === 'se') {
      newBottom = Math.round(toAbsY(clamp(yPct, toPctY(b.top) + minPctY, 100)))
    }
  }

  box.value = {left: newLeft, top: newTop, right: newRight, bottom: newBottom}
  emit('update:range', box.value)
}

function onUp() {
  dragging.value = false
  activeHandle.value = null
  document.removeEventListener('mousemove', onMove)
  document.removeEventListener('mouseup', onUp)
}

const handles: Handle[] = ['nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w']
</script>

<template>
  <div
      class="bbox-container"
      @mousedown="onBodyDown"
  >
    <div
        class="bbox-box"
        :style="{
        left: toPctX(box.left) + '%',
        top: toPctY(box.top) + '%',
        width: toPctX(box.right - box.left) + '%',
        height: toPctY(box.bottom - box.top) + '%',
      }"
    >
      <div
          v-for="h in handles"
          :key="h"
          :class="['bbox-handle', getHandleClass(h)]"
          :style="{ cursor: getCursor(h) }"
          @mousedown.stop="(e: MouseEvent) => onHandleDown(e, h)"
      />
    </div>
  </div>
</template>

<style scoped>
.bbox-container {
  position: absolute;
  inset: 0;
}

.bbox-box {
  position: absolute;
  border: 2px solid #ff4d4f;
  background: rgba(255, 77, 79, 0.06);
  box-sizing: border-box;
}

.bbox-handle {
  position: absolute;
  width: 14px;
  height: 14px;
  background: #fff;
  border: 2px solid #ff4d4f;
  box-sizing: border-box;
  z-index: 2;
}

.bbox-handle:hover {
  background: #ff4d4f;
}

.handle-nw {
  top: -7px;
  left: -7px;
}

.handle-n {
  top: -7px;
  left: calc(50% - 7px);
}

.handle-ne {
  top: -7px;
  right: -7px;
}

.handle-e {
  top: calc(50% - 7px);
  right: -7px;
}

.handle-se {
  bottom: -7px;
  right: -7px;
}

.handle-s {
  bottom: -7px;
  left: calc(50% - 7px);
}

.handle-sw {
  bottom: -7px;
  left: -7px;
}

.handle-w {
  top: calc(50% - 7px);
  left: -7px;
}
</style>
