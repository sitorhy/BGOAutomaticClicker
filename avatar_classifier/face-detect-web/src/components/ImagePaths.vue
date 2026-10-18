<script setup lang="ts">
import {computed, ref} from 'vue'
import {NButton, NIcon, useMessage} from 'naive-ui'
import {CheckCircleFilled} from '@vicons/material'
import {useStore} from '../store/useStore'
import DetectDialog from './DetectDialog.vue'

const store = useStore()
const message = useMessage()

const images = computed(() => store.state.images)
const selectedImage = computed(() => store.state.selectedImage)
const selectedServantId = computed(() => store.state.selectedServantId)
const currentRange = computed(() => store.state.range)

const dialogShow = ref(false)
const dialogIllustration = ref('')

function isMarked(name: string): boolean {
  if (selectedServantId.value == null) return false
  return `${selectedServantId.value}/${name}` in store.state.avatarMap
}

function onImageClick(name: string) {
  store.selectImage(name)
}

function onNew() {
  store.createNewRange()
}

async function onSave() {
  await store.saveRange()
  message.success('已保存')
}

function onDetect(name: string) {
  dialogIllustration.value = name
  dialogShow.value = true
}

function onDetectResult(result: { illustration: string; avatar: string | null; range: any } | null) {
  if (!result || !result.range) {
    message.warning('探测失败：未检测到头像范围')
    return
  }
  const r = result.range
  const [left, top, right, bottom] = r.rect
  store.updateRange({left, top, right, bottom})
  message.success(`探测成功 (score: ${r.score?.toFixed(2) ?? '-'})`)
}
</script>

<template>
  <div class="image-paths">
    <div class="image-paths-title">
      <span>立绘</span>
    </div>
    <div class="image-paths-header">
      <span class="col-name">文件名</span>
      <span class="col-mark">标记</span>
      <span class="col-action">操作</span>
    </div>
    <div class="image-paths-body">
      <div
          v-for="img in images"
          :key="img"
          class="image-path-item"
          :class="{ active: img === selectedImage }"
          @click="onImageClick(img)"
      >
        <span class="col-name" :title="img">{{ img }}</span>
        <span class="col-mark">
          <NIcon v-if="isMarked(img)" size="14" color="#52c41a">
            <CheckCircleFilled/>
          </NIcon>
        </span>
        <span class="col-action">
          <template v-if="img === selectedImage">
            <NButton size="tiny" type="primary" @click.stop="onNew">新建</NButton>
            <NButton size="tiny" type="primary" :disabled="!currentRange" @click.stop="onSave">保存</NButton>
          </template>
          <NButton size="tiny" type="info" @click.stop="onDetect(img)">探测</NButton>
        </span>
      </div>
      <div v-if="images.length === 0" class="image-paths-empty">
        请先选择从者
      </div>
    </div>
    <DetectDialog
        v-if="selectedServantId != null"
        v-model:show="dialogShow"
        :servant-id="selectedServantId"
        :illustration-name="dialogIllustration"
        @detect="onDetectResult"
    />
  </div>
</template>

<style scoped>
.image-paths {
  display: flex;
  flex-direction: column;
  height: 100%;
  background: #fff;
  border-left: 1px solid #e0e0e0;
}

.image-paths-title {
  padding: 12px;
  font-weight: 600;
  font-size: 14px;
  border-bottom: 1px solid #f0f0f0;
}

.image-paths-header {
  display: flex;
  padding: 4px 12px;
  font-size: 11px;
  color: #999;
  border-bottom: 1px solid #f5f5f5;
  background: #fafafa;
}

.image-paths-body {
  flex: 1;
  overflow: auto;
}

.image-path-item {
  display: flex;
  align-items: center;
  padding: 5px 12px;
  font-size: 12px;
  cursor: pointer;
}

.image-path-item:hover {
  background: #f5f5f5;
}

.image-path-item.active {
  background: #e6f4ff;
  font-weight: 500;
}

.col-name {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: #555;
}

.image-path-item.active .col-name {
  color: #1677ff;
}

.col-mark {
  width: 42px;
  text-align: center;
  flex-shrink: 0;
}

.col-action {
  width: 160px;
  display: flex;
  gap: 4px;
  justify-content: center;
  flex-shrink: 0;
}

.image-paths-empty {
  padding: 20px 12px;
  text-align: center;
  color: #bbb;
  font-size: 13px;
}
</style>
