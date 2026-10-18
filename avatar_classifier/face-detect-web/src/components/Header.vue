<script setup lang="ts">
import {ref} from 'vue'
import {NButton, NDropdown, useMessage} from 'naive-ui'
import {ApiManager} from '../api'
import {useStore} from '../store/useStore'

const store = useStore()
const message = useMessage()

const jsonInputRef = ref<HTMLInputElement | null>(null)
const csvInputRef = ref<HTMLInputElement | null>(null)

function triggerFileInput(type: 'json' | 'csv') {
  const input = type === 'json' ? jsonInputRef.value : csvInputRef.value
  input?.click()
}

async function onFileSelected(e: Event, type: 'json' | 'csv') {
  const input = e.target as HTMLInputElement
  const file = input.files?.[0]
  input.value = ''
  if (!file) return

  try {
    const res = type === 'json'
        ? await ApiManager.servantsService.importJson(file)
        : await ApiManager.servantsService.importCsv(file)
    const {imported, total} = res.data
    message.success(`导入完成：${imported} / ${total}`)
    await store.init()
  } catch (err: any) {
    message.error(`导入失败：${err.message ?? err}`)
  }
}

const options = [
  {label: '批量导入从者数据[JSON]', key: 'import-json'},
  {label: '批量导入从者数据[CSV]', key: 'import-csv'},
]

function handleSelect(key: string | number) {
  switch (key) {
    case 'import-json':
      triggerFileInput('json')
      break
    case 'import-csv':
      triggerFileInput('csv')
      break
  }
}
</script>

<template>
  <div class="header">
    <div class="header-title">头像探测</div>
    <div class="header-shortcuts">
      <NDropdown trigger="hover" :options="options" @select="handleSelect">
        <NButton>操作</NButton>
      </NDropdown>
    </div>
    <input
        ref="jsonInputRef"
        type="file"
        accept=".json"
        style="display: none"
        @change="(e: Event) => onFileSelected(e, 'json')"
    />
    <input
        ref="csvInputRef"
        type="file"
        accept=".csv"
        style="display: none"
        @change="(e: Event) => onFileSelected(e, 'csv')"
    />
  </div>
</template>

<style scoped>
.header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 20px;
  background: #fff;
  border-bottom: 1px solid #e0e0e0;
}

.header-title {
  font-size: 18px;
  font-weight: 600;
  color: #333;
}

.header-shortcuts {
  display: flex;
  gap: 16px;
  font-size: 12px;
  color: #999;
}
</style>
